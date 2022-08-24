// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_common.hpp"
#include "common_utils.hpp"
#include "test_constants.hpp"

#include <threading/ie_executor_manager.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <random>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define _WINSOCKAPI_

#include <windows.h>
#include "Psapi.h"
#endif

#if ENABLE_CONFORMANCE_PGQL
#include <sstream>
#include <chrono>
#include "libpq-fe.h"
#define PGQL_DEBUG
#undef PGQL_DEBUG
static const char* PGQL_ENV_CONN_NAME = "OV_POSTGRES_CONN";  //Environment variable with connection settings
static const char* PGQL_ENV_SESS_NAME = "OV_TEST_SESSION_ID";//Environment variable identifies current session
#endif

namespace CommonTestUtils {

inline size_t getVmSizeInKB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
        pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb);
        return pmc.WorkingSetSize;
#else
    auto parseLine = [](char *line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char *p = line;
        while (*p < '0' || *p > '9') p++;
        line[i - 3] = '\0';
        i = (size_t) atoi(p);
        return i;
    };

    FILE *file = fopen("/proc/self/status", "r");
    size_t result = 0;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
    }
    return result;
#endif
}

TestsCommon::~TestsCommon() {
    InferenceEngine::executorManager()->clear();
}

TestsCommon::TestsCommon() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
    InferenceEngine::executorManager()->clear();
}

std::string TestsCommon::GetTimestamp() {
    return CommonTestUtils::GetTimestamp();
}

std::string TestsCommon::GetTestName() const {
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(test_name.begin(), test_name.end(),
        [](char c) { return !std::isalnum(c); }, '_');
    return test_name;
}

/*
    PostgreSQL Handler class members
*/
#ifdef PGQL_DEBUG
#define SAY_HELLO std::cout << "Hi folks, " << __FUNCTION__ << std::endl;
#else
#define SAY_HELLO 
#endif
/*
    This class implements singleton which operates with a connection to PostgreSQL server.
*/
class PostgreSQLConnection {
    PGconn* activeConnection;

    PostgreSQLConnection() : activeConnection(nullptr), isConnected(false) {}

    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLConnection(const PostgreSQLConnection&) = delete;
    PostgreSQLConnection& operator=(const PostgreSQLConnection&) = delete;

    /* This destructor is using for a making correct removal of PGresult object */
    class PGresultDeleter {
    public:
        void operator()(PGresult* ptr) {
            if (!ptr)
                return;
            PQclear(ptr);
        }
    };

public:
    bool isConnected;

    static PostgreSQLConnection& GetInstance(void);
    bool Initialize();
    /* Queries a server. Result will be returned as self-desctructable pointer. But application should check result pointer
    isn't a nullptr. */
    std::shared_ptr<PGresult> Query(const char* query) {
        if (!isConnected)
            return std::shared_ptr<PGresult>(nullptr, PGresultDeleter{});
        return std::shared_ptr<PGresult>(PQexec(this->activeConnection, query), PGresultDeleter{});
    }
    ~PostgreSQLConnection();
};

PostgreSQLConnection& PostgreSQLConnection::GetInstance(void) {
    static PostgreSQLConnection connection;
    return connection;
}

PostgreSQLConnection::~PostgreSQLConnection() {
    if (activeConnection) {
        PQfinish(this->activeConnection);
        this->activeConnection = nullptr;
        isConnected = false;
    }
}

/*
    Initialization of exact object. Uses environment variable PGQL_ENV_CONN_NAME for making a connection.
    Returns false in case of failure or absence of ENV-variable.
    Returns true in case of connection has been succesfully established.
*/
bool PostgreSQLConnection::Initialize() {
    if (this->activeConnection != nullptr) {
        std::cerr << "PostgreSQL connection already established." << std::endl;
        return true;
    }

    const char* envConnString = nullptr;
    envConnString = std::getenv(PGQL_ENV_CONN_NAME);

    if (envConnString == nullptr) {
        std::cerr << "PostgreSQL connection string isn't found in Environment (" << PGQL_ENV_CONN_NAME << ")" << std::endl;
        return false;
    } else {
        std::cerr << "PostgreSQL connection string: " << envConnString;
    }

    this->activeConnection = PQconnectdb(envConnString);

    ConnStatusType connStatus = PQstatus(this->activeConnection);

    if (connStatus != CONNECTION_OK) {
        std::cerr << "Cannot connect to PostgreSQL: " << static_cast<uint32_t>(connStatus) << std::endl;
        return false;
    } else {
        std::cerr << "Connected to PostgreSQL successfully" << std::endl;
    }

    this->isConnected = true;

    return true;
}

#define CHECK_PGQUERY(var_name)        \
    if (var_name.get() == nullptr) { \
        std::cerr << "Error while querying PostgreSQL" << std::endl; \
        return; \
    }

class PostgreSQLEventListener : public ::testing::EmptyTestEventListener {
    const char* session_id = nullptr;
    bool isPostgresEnabled = false;

    /* Dynamic information about current session*/
    uint64_t sessionId = 0;
    uint64_t testIterationId = 0;
    uint64_t testSuiteNameId = 0;
    uint64_t testNameId = 0;
    uint64_t testSuiteId = 0;
    uint64_t testId = 0;

    void OnTestProgramStart(const ::testing::UnitTest& /*unit_test*/) override {
    }
    void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {
        std::stringstream sstr;
        sstr << "INSERT INTO test_iterations (name, iteration) VALUES ('" << unit_test.current_test_suite()->name() << "', "
             << iteration << ")";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
    }
    void OnEnvironmentsSetUpStart(const ::testing::UnitTest& /*unit_test*/) override {
        SAY_HELLO;
    }
    void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& /*unit_test*/) override {
        SAY_HELLO;
    }
    void OnTestSuiteStart(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "SELECT GET_TEST_SUITE('" << test_suite.name() << "')";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
        auto pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        ExecStatusType execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_TUPLES_OK) {
            std::cerr << "Cannot retrieve a correct sn_id, error: " << static_cast<uint64_t>(execStatus)
                      << std::endl;
        }
        this->testSuiteNameId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
        if (this->testSuiteNameId == 0) {
            std::cerr << "Cannot interpret a returned sn_id, value: " << PQgetvalue(pgresult.get(), 0, 0)
                      << std::endl;
        }

        sstr.str(std::string());
        sstr << "INSERT INTO suite_results (sr_id, session_id, suite_id) VALUES (DEFAULT, " << this->sessionId << ", "
             << this->testSuiteNameId << ") RETURNING sr_id";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_TUPLES_OK) {
            std::cerr << "Cannot retrieve a correct sr_id, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testSuiteId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
        if (this->testSuiteId == 0) {
            std::cerr << "Cannot interpret a returned sr_id, value: " << PQgetvalue(pgresult.get(), 0, 0) << std::endl;
        }
    }
//  Legacy API is deprecated but still available
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    void OnTestCaseStart(const ::testing::TestCase& test_case) override {
        if (this->testSuiteNameId == 0)
            OnTestSuiteStart(test_case);
    }
#endif  //  GTEST_REMOVE_LEGACY_TEST_CASEAPI_

    void OnTestStart(const ::testing::TestInfo& test_info) override {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "SELECT GET_TEST_NAME('" << test_info.name() << "', " << this->testSuiteNameId << ")";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
        auto pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        ExecStatusType execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_TUPLES_OK) {
            std::cerr << "Cannot retrieve a correct tn_id, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testNameId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
        if (this->testNameId == 0) {
            std::cerr << "Cannot interpret a returned tn_id, value: " << PQgetvalue(pgresult.get(), 0, 0) << std::endl;
        }

        sstr.str(std::string());
        sstr << "INSERT INTO test_results (tr_id, session_id, suite_id, test_id) VALUES (DEFAULT, " << this->sessionId << ", " << this->testSuiteId << ", " << this->testNameId << ") RETURNING tr_id";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_TUPLES_OK) {
            std::cerr << "Cannot retrieve a correct tr_id, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
        if (this->testId == 0) {
            std::cerr << "Cannot interpret a returned tr_id, value: " << PQgetvalue(pgresult.get(), 0, 0) << std::endl;
        }
    }
    void OnTestPartResult(const ::testing::TestPartResult& test_part_result) override {
        std::stringstream sstr;
        sstr << "INSERT INTO test_starts(part) (name) VALUES (\"partresult\")";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
    }
    void OnTestEnd(const ::testing::TestInfo& test_info) override {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        uint32_t testResult = 0;
        if (test_info.result()->Passed())
            testResult = 1;
        else if (test_info.result()->Skipped())
            testResult = 2;
        sstr << "UPDATE test_results SET finished_at=NOW(), duration=" << test_info.result()->elapsed_time()
             << ", test_result=" << testResult << " WHERE tr_id=" << this->testId;
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
        auto pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        ExecStatusType execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_COMMAND_OK) {
            std::cerr << "Cannot update test results, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testId = 0;
    }
    void OnTestSuiteEnd(const ::testing::TestSuite& test_suite) override {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "UPDATE suite_results SET finished_at=NOW(), duration=" << test_suite.elapsed_time()
             << ", suite_result=" << (test_suite.Passed() ? 1 : 0) << " WHERE sr_id=" << this->testSuiteId;
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
        auto pgresult = conn.Query(sstr.str().c_str());
        CHECK_PGQUERY(pgresult);
        ExecStatusType execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_COMMAND_OK) {
            std::cerr << "Cannot update test suite results, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testSuiteId = 0;
    }
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
    void OnTestCaseEnd(const ::testing::TestCase& test_case) override {
        if (this->testSuiteId != 0)
            OnTestSuiteEnd(test_case);
    }
#endif  //  GTEST_REMOVE_LEGACY_TEST_CASEAPI_

    void OnEnvironmentsTearDownStart(const ::testing::UnitTest& /*unit_test*/) override {
        SAY_HELLO;
    }
    void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& /*unit_test*/) override {
        SAY_HELLO;
    }
    void OnTestIterationEnd(const ::testing::UnitTest& /*unit_test*/, int /*iteration*/) override {
        std::stringstream sstr;
        sstr << "UPDATE test_iterations WHERE id=... SET finished=\"finishdate\")";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
    }
    void OnTestProgramEnd(const ::testing::UnitTest& /*unit_test*/) override {
    }
    /* Do nothing here. If you need to do anything on creation - it should be fully undersandable. */
    PostgreSQLEventListener() {
        this->session_id = std::getenv(PGQL_ENV_SESS_NAME);
        if (this->session_id != nullptr) {
            isPostgresEnabled = false;

            std::cerr << "Test session ID has been found" << std::endl;
            PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
            bool connInitResult = conn.Initialize();

            if (!connInitResult)
                return;

            std::stringstream sstr;
            sstr << "SELECT GET_SESSION(" << this->session_id << ")";
#ifdef PGQL_DEBUG
            std::cerr << sstr.str() << std::endl;
#endif
            auto pgresult = conn.Query(sstr.str().c_str());
            CHECK_PGQUERY(pgresult);

            ExecStatusType execStatus = PQresultStatus(pgresult.get());
            isPostgresEnabled = connInitResult;
            if (execStatus != PGRES_TUPLES_OK) {
                std::cerr << "Cannot retrieve a correct session_id, error: " << static_cast<uint64_t>(execStatus)
                          << std::endl;
                isPostgresEnabled = false;
            }
            this->sessionId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
            if (this->sessionId == 0) {
                std::cerr << "Cannot interpret a returned session_id, value: " << PQgetvalue(pgresult.get(), 0, 0)
                          << std::endl;
                isPostgresEnabled = false;
            }
        } else {
            std::cerr << "Test session ID hasn't been found, continues without database reporting" << std::endl;
        }
    }
    ~PostgreSQLEventListener() {
        if (!this->isPostgresEnabled)
            return;

        std::stringstream sstr;
        sstr << "UPDATE sessions SET end_time=NOW() WHERE session_id=" << this->sessionId << " AND end_time<NOW()";
#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();
        auto pgresult = conn.Query(sstr.str().c_str());
        ExecStatusType execStatus = PQresultStatus(pgresult.get());
        if (execStatus != PGRES_COMMAND_OK) {
            std::cerr << "Cannot update session finish info, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
    }
    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLEventListener(const PostgreSQLEventListener&) = delete;
    PostgreSQLEventListener& operator=(const PostgreSQLEventListener&) = delete;

public:
    static PostgreSQLEventListener* GetListener() {
        static PostgreSQLEventListener listener;
        return &listener;
    }
};

class PostgreSQLEnvironment : public ::testing::Environment {
public:
    PostgreSQLEnvironment() {
        SAY_HELLO;
    }
    ~PostgreSQLEnvironment() {
        SAY_HELLO;
    }
    virtual void SetUp() {
        SAY_HELLO;
        if (std::getenv(PGQL_ENV_SESS_NAME) != nullptr && std::getenv(PGQL_ENV_CONN_NAME) != nullptr) {
            ::testing::UnitTest::GetInstance()->listeners().Append(PostgreSQLEventListener::GetListener());
        }
    }
    virtual void TearDown() {
        SAY_HELLO;
    }
};

::testing::Environment *PostgreSQLEnvironment_Reg = ::testing::AddGlobalTestEnvironment(new PostgreSQLEnvironment());

PostgreSQLHandler::PostgreSQLHandler() {
    std::cout << "PostgreSQLHandler Started\n";

    ::testing::Test* gTest = nullptr;
    TestsCommon* tcTest = nullptr;

    try {
        gTest = dynamic_cast<::testing::Test*>(this);
//        std::cout << "Got gTest!\n";
    }
    catch(std::bad_cast) {
        std::cout << "Cannot cast this to Test*\n";
    }

    try {
        tcTest = dynamic_cast<TestsCommon*>(this);
//        std::cout << "Got tcTest! Test name is " << tcTest->GetTestName()
//                  << "\n";
    } catch (std::bad_cast) {
        std::cout << "Cannot cast this to TestsCommon*\n";
    }
}

PostgreSQLHandler::~PostgreSQLHandler() {
    ::testing::Test* gTest = nullptr;
    TestsCommon* tcTest = nullptr;

    try {
        gTest = dynamic_cast<::testing::Test*>(this);
//        std::cout << "Got gTest!\n";
    } catch (std::bad_cast) {
        std::cout << "Cannot cast this to Test*\n";
    }

    try {
        tcTest = dynamic_cast<TestsCommon*>(this);
//        std::cout << "Got tcTest! Test name is " << tcTest->GetTestName() << "\n";
    } catch (std::bad_cast) {
        std::cout << "Cannot cast this to TestsCommon*\n";
    }

    if (gTest) {
        if (gTest->HasFailure())
            std::cout << "Failure\n";
        else if (gTest->IsSkipped())
            std::cout << "Skipped\n";
        else
            std::cout << "Success!\n";    
    }
    std::cout << "PostgreSQLHandler Finished\n";
}

void PostgreSQLHandler::SetUp() {
    SAY_HELLO;
}

void PostgreSQLHandler::TearDown() {
    SAY_HELLO;
}

}  // namespace CommonTestUtils
