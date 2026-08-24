from http.client import HTTPException

from pymysql import IntegrityError
from sqlalchemy.exc import SQLAlchemyError

from utils.exception import http_exception_handler, general_error_handler, sqlalchemy_error_handler, integrity_error_handler


def register_exception_handler(app):
    """
    注册全局异常处理: 子类在前，父类在后；具体在前，抽象在后
    """
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(IntegrityError, integrity_error_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)
    app.add_exception_handler(Exception, general_error_handler)
