from passlib.context import CryptContext

# 创建密码上下文
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

# 密码加密
def get_hash_password(password: str):
    return pwd_context.hash(password)

# 验证密码: 明文，密文
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

