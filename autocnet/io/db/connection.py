from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, pool, orm
from sqlalchemy.orm import create_session, scoped_session, sessionmaker

import os
import socket
import warnings
import yaml


class Parent:
    def __init__(self, config):
        self.session, _ = new_connection(config['database'])
        self.session.begin()

def new_connection(db):
    """
    Using the user supplied config create a NullPool database connection.

    Parameters
    ----------
    db : dict
         in the form {'username':username, 'password': password,
                      'host':db hostname, 'pgbouncer_port': database port,
                      'name': database name}

    Returns
    -------
    Session : object
              An SQLAlchemy session object

    engine : object
             An SQLAlchemy engine object
    """
    db_uri = 'postgresql://{}:{}@{}:{}/{}'.format(db['username'],
                                                  db['password'],
                                                  db['host'],
                                                  db['pgbouncer_port'],
                                                  db['name'])    
    engine = sqlalchemy.create_engine(db_uri,
                                      poolclass=sqlalchemy.pool.NullPool)
    Session = sqlalchemy.orm.sessionmaker(bind=engine, autocommit=True)
    return Session, engine
