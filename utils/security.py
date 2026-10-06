"""
密码哈希。

为什么不使用 passlib：
passlib 最后一次发版是 2020 年（1.7.4），已停止维护。它在探测 bcrypt 后端时会用
一个 73 字节的探针密码做兼容性检测，而 bcrypt 4.1+ 明确禁止超过 72 字节的密码，
于是 passlib 1.7.4 + bcrypt 4.1+ 组合会直接抛
    ValueError: password cannot be longer than 72 bytes
注册接口 500。原生 bcrypt 没有这个问题，也没有这层多余抽象。

为什么用 bcrypt 而不是 md5/sha256：
md5/sha256 是为速度设计的哈希，可被彩虹表秒破，且没有盐。
bcrypt 专门为密码设计：自带随机盐、计算成本可调（故意算得慢），
即使数据库泄露，爆破成本也极高。rounds=12 是当前常用的安全/性能平衡点。
"""
import base64
import hashlib

import bcrypt

# 计算成本：每 +1 耗时翻倍。12 约 0.25s/次，可按服务器性能微调
BCRYPT_ROUNDS = 12

# bcrypt 原生限制：密码超过 72 字节会报错
BCRYPT_MAX_BYTES = 72


def _prepare(password: str) -> bytes:
    """
    把密码转成 bcrypt 能吃的 bytes。

    超过 72 字节时先做一次 sha256 摘要再 base64，而不是直接截断——
    截断会让「abcdef...72位」和「abcdef...73位」得到相同哈希，形成安全隐患。
    """
    raw = password.encode("utf-8")
    if len(raw) > BCRYPT_MAX_BYTES:
        raw = base64.b64encode(hashlib.sha256(raw).digest())
    return raw


def get_hash_password(password: str) -> str:
    """生成密码哈希，返回可直接存库的字符串"""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(_prepare(password), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    校验明文密码是否与哈希匹配。

    哈希格式非法或为空时返回 False 而不是抛异常——调用方（登录接口）
    不应该因为库里一个脏数据就返回 500。
    """
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(_prepare(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False