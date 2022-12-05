// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "postgres_link.hpp"

#include <gtest/gtest.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <map>
#include <pugixml.hpp>
#include <sstream>

//#define PGQL_DYNAMIC_LOAD
/// \brief Enables extended debug messages to the stderr
#define PGQL_DEBUG
//#undef PGQL_DEBUG
static const char* PGQL_ENV_CONN_NAME = "OV_POSTGRES_CONN";    // Environment variable with connection settings
static const char* PGQL_ENV_SESS_NAME = "OV_TEST_SESSION_ID";  // Environment variable identifies current session

#ifndef PGQL_DYNAMIC_LOAD
#include "libpq-fe.h"
#endif

namespace CommonTestUtils {

/*
    PostgreSQL Handler class members
*/
#ifdef PGQL_DEBUG
#    define SAY_HELLO std::cout << "Hi folks, " << __FUNCTION__ << std::endl;
#else
#    define SAY_HELLO \
        {}
#endif

/* This manager is using for a making correct removal of PGresult object.
shared/unique_ptr cannot be used due to incomplete type of PGresult.
It is minimal implementatio which is compatible with shared/uinque_ptr
interface usage (reset, get) */
class PGresultHolder {
    PGresult* _ptr;
    volatile uint32_t* refCounter;

    inline void decRefCounter() {
        if (_ptr != nullptr && refCounter != nullptr) {
            if (*refCounter > 0) {
                --*refCounter;
            }
            if (*refCounter == 0) {
                delete refCounter;
                PQclear(_ptr);
                _ptr = nullptr;
            }
        }
    }

public:
    PGresultHolder() : _ptr(nullptr), refCounter(nullptr) {}
    PGresultHolder(PGresult* ptr) : _ptr(ptr), refCounter(new uint32_t()) {
        *refCounter = 1;
    }
    PGresultHolder(const PGresultHolder& object) {
        _ptr = object._ptr;
        refCounter = object.refCounter;
        ++*refCounter;
    }
    PGresultHolder& operator=(const PGresultHolder& object) {
        if (_ptr != object._ptr) {
            decRefCounter();
            _ptr = object._ptr;
            refCounter = object.refCounter;
            ++*refCounter;
        }
        return *this;
    }
    void reset(PGresult* ptr) {
        if (_ptr != ptr) {
            decRefCounter();
            if (ptr != nullptr && refCounter == nullptr) {
                refCounter = new uint32_t();
                *refCounter = 1;
            }
            _ptr = ptr;
        }
    }
    PGresult* get() {
        return _ptr;
    }
    ~PGresultHolder() {
        decRefCounter();
        _ptr = nullptr;
        refCounter = nullptr;
    }
};

/*
    This class implements singleton which operates with a connection to PostgreSQL server.
*/
class PostgreSQLConnection {
    PGconn* activeConnection;

    PostgreSQLConnection() : activeConnection(nullptr), isConnected(false) {}

    /* Prohobit creation outsize of class, need to make a Singleton */
    PostgreSQLConnection(const PostgreSQLConnection&) = delete;
    PostgreSQLConnection& operator=(const PostgreSQLConnection&) = delete;

public:
    bool isConnected;

    static PostgreSQLConnection& GetInstance(void);
    bool Initialize();
    /* Queries a server. Result will be returned as self-desctructable pointer. But application should check result
    pointer isn't a nullptr. */
    PGresultHolder Query(const char* query) {
        if (!isConnected)
            return PGresultHolder();
        PGresultHolder result(PQexec(this->activeConnection, query));
        // Connection could be closed by a timeout, we may try to reconnect once.
        // We don't reconnect on each call because it may make testing significantly slow in
        // case of connection issues. Better to finish testing with incomplete results and
        // free a machine. Otherwise we will lose all results.
        if (result.get() == nullptr) {
            TryReconnect();
            // If reconnection attempt was successfull - let's try to set new query
            if (isConnected) {
                result.reset(PQexec(this->activeConnection, query));
            }
        }
        return result;
    }

    void TryReconnect(void) {
        if (!isConnected) {
            return;
        }
        if (activeConnection != nullptr) {
            try {
                PQfinish(activeConnection);
            } catch (...) {
                std::cerr << "An exception while finishing PostgreSQL connection" << std::endl;
            }
            this->activeConnection = nullptr;
            this->isConnected = false;
        }
        std::cerr << "Reconnecting to the PostgreSQL server..." << std::endl;
        Initialize();
    }

    PGconn* GetConnection(void) {
        return this->activeConnection;
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
        this->isConnected = false;
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
        std::cerr << "PostgreSQL connection string isn't found in Environment (" << PGQL_ENV_CONN_NAME << ")"
                  << std::endl;
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

#define CHECK_PGQUERY(var_name)                                      \
    if (var_name.get() == nullptr) {                                 \
        std::cerr << "Error while querying PostgreSQL" << std::endl; \
        return;                                                      \
    }

/*
    This is a workaround to place all data in one cpp file.

    In production version it has to be separated for heeader and source
    and this part should be removed;

*/

/*
    Known issues:
    - String escape isn't applied for all fields (PoC limitation)
*/
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
    std::map<std::string, std::string> testCustomFields;

    /*
    This method is used for parsing serialized value_param string.

    Known limitations:
    It doesn't read values in inner tuples/arrays/etc.
    */
    std::vector<std::string> ParseValueParam(std::string text) {
        std::vector<std::string> results;
        size_t beginning = 0;
        size_t chrPos = 0;
        char pairingChar = 0;
        for (auto it = text.begin(); it != text.end(); ++it, ++chrPos) {
            if (pairingChar == 0) {  // Looking for opening char
                switch (*it) {
                case '"':
                case '\'':
                    pairingChar = *it;
                    break;
                case '{':
                    pairingChar = '}';
                    break;
                }
                beginning = chrPos + 1;
            } else if (*it != pairingChar) {  // Skip while don't face with paring char
                continue;
            } else {
                if (chrPos < 3 || (text[chrPos - 1] != '\\' && text[chrPos - 2] != '\\')) {
                    size_t substrLength = chrPos - beginning;
                    if (substrLength > 0 && (beginning + substrLength) < text.length()) {
                        results.push_back(text.substr(beginning, chrPos - beginning));
                    }
                    pairingChar = 0;
                }
            }
        }
        return results;
    }
    /*
        void OnTestProgramStart(const ::testing::UnitTest& unit_test) override {
        }
        void OnTestIterationStart(const ::testing::UnitTest& unit_test, int iteration) override {
            std::stringstream sstr;
            sstr << "INSERT INTO test_iterations (name, iteration) VALUES ('" << unit_test.current_test_suite()->name()
    << "', "
                 << iteration << ")";
    #ifdef PGQL_DEBUG
            std::cerr << sstr.str() << std::endl;
    #endif
        }
        void OnEnvironmentsSetUpStart(const ::testing::UnitTest& unit_test) override {
            SAY_HELLO;
        }
        void OnEnvironmentsSetUpEnd(const ::testing::UnitTest& unit_test) override {
            SAY_HELLO;
        }
    */
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
            std::cerr << "Cannot retrieve a correct sn_id, error: " << static_cast<uint64_t>(execStatus) << std::endl;
        }
        this->testSuiteNameId = std::atoi(PQgetvalue(pgresult.get(), 0, 0));
        if (this->testSuiteNameId == 0) {
            std::cerr << "Cannot interpret a returned sn_id, value: " << PQgetvalue(pgresult.get(), 0, 0) << std::endl;
        }

        sstr.str("");
        sstr.clear();
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

        PostgreSQLConnection& conn = PostgreSQLConnection::GetInstance();

        std::stringstream sstr;
        sstr << "SELECT GET_TEST_NAME(" << this->testSuiteNameId << ", '" << test_info.name() << "'";
        /*
            This part might be specific for different tests. In case amount of cases will be greater than, for example,
           2 the code should be refactored to use a map on test-dependent functions/methods.
        */
        if (test_info.value_param() != NULL &&
            strcmp(::testing::UnitTest::GetInstance()->current_test_suite()->name(), "conformance/ReadIRTest") == 0) {
            /*
            This part of code responsible for cleaning source model XML from
            meaningless information which might be changed run2run by SubgraphDumper
            or similar tool
            */
            std::string testDescription;
            {
                std::ostringstream normalizedXml;
                std::vector<std::string> params =
                    ParseValueParam(test_info.value_param());  // Extracting value_params from serialized string

                if (params.size() > 0) {
                    struct modelNormalizer : pugi::xml_tree_walker {
                        bool for_each(pugi::xml_node& node) override {
                            if (node.type() == pugi::node_element) {
                                node.remove_attribute("name");
                            }
                            return true;
                        }
                    } modelNorm;

                    pugi::xml_document normalizedModel;
                    normalizedModel.load_file(params[0].c_str());
                    normalizedModel.traverse(modelNorm);
                    normalizedModel.save(normalizedXml, "");  // No indent expected, let's reduce size of xml
                    testDescription = normalizedXml.str();
                }
            }
            if (!testDescription.empty()) {
                /*
                    A generated XML may contains characters should be escaped in a query.
                */
                std::vector<char> escapedDescription;
                escapedDescription.resize(testDescription.length() *
                                          2);  // Doc requires to allocate two times more than initial length
                escapedDescription[0] = 0;     //
                int errCode = 0;
                size_t writtenSize = 0;
                writtenSize = PQescapeStringConn(conn.GetConnection(),
                                                 escapedDescription.data(),
                                                 testDescription.c_str(),
                                                 testDescription.length(),
                                                 &errCode);
                if (writtenSize >= testDescription.length()) {
                    sstr << ", '" << std::string(escapedDescription.data()) << "'";
                } else {
                    std::cerr << "Cannot escape string (error code is " << errCode << "):" << std::endl
                              << testDescription;
                }
            }
        }
        sstr << ")";

#ifdef PGQL_DEBUG
        std::cerr << sstr.str() << std::endl;
#endif
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

        sstr.str("");
        sstr.clear();
        sstr << "INSERT INTO test_results (tr_id, session_id, suite_id, test_id) VALUES (DEFAULT, " << this->sessionId
             << ", " << this->testSuiteId << ", " << this->testNameId << ") RETURNING tr_id";
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
        /*
            void OnEnvironmentsTearDownStart(const ::testing::UnitTest& unit_test) override {
                SAY_HELLO;
            }
            void OnEnvironmentsTearDownEnd(const ::testing::UnitTest& unit_test) override {
                SAY_HELLO;
            }
            void OnTestIterationEnd(const ::testing::UnitTest& unit_test, int iteration) override {
                std::stringstream sstr;
                sstr << "UPDATE test_iterations WHERE id=... SET finished=\"finishdate\")";
        #ifdef PGQL_DEBUG
                std::cerr << sstr.str() << std::endl;
        #endif
            }
            void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
            }
        */
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

    friend class PostgreSQLEnvironment;

public:
    bool SetCustomField(const std::string fieldName, const std::string fieldValue, const bool rewrite) {
        auto field = this->testCustomFields.find(fieldName);
        if (rewrite || field != this->testCustomFields.end()) {
            this->testCustomFields[fieldName] = fieldValue;
            return true;
        }
        return false;
    }

    std::string GetCustomField(const std::string fieldName, const std::string defaultValue) const {
        auto field = this->testCustomFields.find(fieldName);
        if (field != this->testCustomFields.end()) {
            return field->second;
        }
        return defaultValue;
    }

    bool RemoveCustomField(const std::string fieldName) {
        auto field = this->testCustomFields.find(fieldName);
        if (field != this->testCustomFields.end()) {
            this->testCustomFields.erase(field);
            return true;
        }
        return false;
    }

    void ClearCustomFields() {
        this->testCustomFields.clear();
    }
};

static PostgreSQLEventListener* pgEventListener = nullptr;

class PostgreSQLEnvironment : public ::testing::Environment {
public:
    PostgreSQLEnvironment() {
        SAY_HELLO;
    }
    ~PostgreSQLEnvironment() {
        SAY_HELLO;
    }
    void SetUp() override {
        SAY_HELLO;
        if (std::getenv(PGQL_ENV_SESS_NAME) != nullptr && std::getenv(PGQL_ENV_CONN_NAME) != nullptr) {
            if (pgEventListener == nullptr) {
                pgEventListener = new PostgreSQLEventListener();
                ::testing::UnitTest::GetInstance()->listeners().Append(pgEventListener);
            }
        }
    }
    void TearDown() override {
        SAY_HELLO;
    }
};

::testing::Environment* PostgreSQLEnvironment_Reg = ::testing::AddGlobalTestEnvironment(new PostgreSQLEnvironment());

/* This structure is for internal usage, don't need to move it to the header */
class PostgreSQLCustomData {
public:
    std::map<std::string, std::string> customFields;
};

PostgreSQLLink::PostgreSQLLink()
    : parentObject(nullptr),
      customData(nullptr)
{
    this->parentObject = nullptr;
    std::cout << "PostgreSQLLink Started\n";
    this->customData = new PostgreSQLCustomData();
}

PostgreSQLLink::~PostgreSQLLink() {
    if (this->customData) {
        delete this->customData;
        this->customData = nullptr;
    }

    this->parentObject = nullptr;

    std::cout << "PostgreSQLLink Finished\n";
}

bool PostgreSQLLink::SetCustomField(const std::string fieldName,
                                          const std::string fieldValue,
                                          const bool rewrite) {
    if (pgEventListener) {
        if (!pgEventListener->SetCustomField(fieldName, fieldValue, rewrite))
            return false;
    }
    auto field = this->customData->customFields.find(fieldName);
    if (rewrite || field != this->customData->customFields.end()) {
        this->customData->customFields[fieldName] = fieldValue;
        return true;
    }
    return false;
}

std::string PostgreSQLLink::GetCustomField(const std::string fieldName, const std::string defaultValue) const {
    if (pgEventListener) {
        return pgEventListener->GetCustomField(fieldName, defaultValue);
    }
    auto field = this->customData->customFields.find(fieldName);
    if (field != this->customData->customFields.end()) {
        return field->second;
    }
    return defaultValue;
}

bool PostgreSQLLink::RemoveCustomField(const std::string fieldName) {
    if (pgEventListener) {
        pgEventListener->RemoveCustomField(fieldName);
    }
    auto field = this->customData->customFields.find(fieldName);
    if (field != this->customData->customFields.end()) {
        this->customData->customFields.erase(field);
        return true;
    }
    return false;
}

}  // namespace CommonTestUtils
