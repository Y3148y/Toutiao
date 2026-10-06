import os

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine

# 数据库配置：从环境变量读取，未设置时使用本地默认值
# 真实凭据请通过环境变量注入，切勿硬编码进代码
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "news_app")
DB_CHARSET = os.getenv("DB_CHARSET", "utf8mb4")

# 数据库URL
ASYNC_DATABASE_URL = (
    f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset={DB_CHARSET}"
)

# 创建异步引擎
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",  # 输出SQL日志
    pool_size=int(os.getenv("DB_POOL_SIZE", "20")),  # 设置连接池中保持的持久连接数
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10"))  # 连接池溢出连接数
)

# 创建异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    """
    创建依赖项获取数据库会话
    :return:
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session # 异步生成器，把session返回给调用方，直到调用方使用session完成操作后，session才会被释放
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()