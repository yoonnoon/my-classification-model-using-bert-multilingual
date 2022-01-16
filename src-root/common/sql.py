import logging
from datetime import datetime

import sqlalchemy

log = logging.getLogger()


def get_sqlalchemy_url(vendor, user, pwd, host, port, dbname):
    return f"{vendor}://{user}:{pwd}@{host}:{port}/{dbname}"


def select_no_label_qna_contents(url, range_lower__gte, range_upper_lt):
    datetime.strptime(range_lower__gte, "%Y-%m-%d")
    datetime.strptime(range_upper_lt, "%Y-%m-%d")

    query = f"""
    select
        id,
        contents
    from
        sample.qna
    where
        label is null
        and contents is not null
        and timezone('kst', created_at) >= '{range_lower__gte}'
        and timezone('kst', created_at) < '{range_upper_lt}'
    """

    with sqlalchemy.create_engine(url).connect() as connection:
        ret = connection.execute(sqlalchemy.text(query))
        log.info("Query: %s." % query)

        rows = ret.fetchall()

        data = {}
        for row in rows:
            row = dict(row)
            for col, value in row.items():
                if col not in data:
                    data[col] = [value]
                else:
                    data[col].append(value)
        return data


def update_qna_contents_label(url, label, ids):
    ret = 0
    if len(ids) > 0:
        query = f"""
            update
                sample.qna
            set
                label = '{label}'
            where
                id in {tuple(ids) if len(tuple(ids)) > 1 else '(%s)' % ids[0]}
        """
        with sqlalchemy.create_engine(url).begin() as transaction:
            ret = transaction.execute(sqlalchemy.text(query)).rowcount
            log.info("Query: %s, rows: %s." % (query, ret))
    return ret
