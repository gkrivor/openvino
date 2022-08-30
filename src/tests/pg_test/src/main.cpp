#include <stdio.h>
#if ENABLE_CONFORMANCE_PGQL
#include <stdlib.h>
#include <memory>
#include "libpq-fe.h"
#endif

int main(int, char**)
{
#if ENABLE_CONFORMANCE_PGQL
    std::shared_ptr<PGconn> conn;
    std::shared_ptr<PGresult> res;
    ExecStatusType rType;
    const char* conninfo =
        "host=conformancedb.sclab.intel.com port=8080 user=postgres password=Idjfnj834r dbname=postgres";
#endif
    printf("Hello Postgres!\n");

#if ENABLE_CONFORMANCE_PGQL
    printf("Connecting to the server...\n");
    conn = std::shared_ptr<PGconn>(PQconnectdb(conninfo), [](PGconn* connection) {
        if (!connection)
            return;
        printf("PQfinish()\n");
        PQfinish(connection);
    });

    if (PQstatus(conn.get()) != CONNECTION_OK) {
        printf("Cannot connect to the server");
        return 1;
    }

    printf("Querying the server...\n");
    res = std::shared_ptr<PGresult>(PQexec(conn.get(), "select datname from pg_database"), [](PGresult* result) {
        if (!result)
            return;
        printf("PQclear()\n");
        PQclear(result);
    });

    rType = PQresultStatus(res.get());

    if (rType != PGRES_TUPLES_OK) {
        printf("Cannot fetch data %d, %s", int(rType), PQresultErrorMessage(res.get()));
        return 2;
    }

    for (int i = 0; i < PQntuples(res.get()); ++i) {
        printf("%s\n", PQgetvalue(res.get(), i, 0));
    }
#endif

    return 0;
}