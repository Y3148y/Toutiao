from click import echo
from pydantic.v1.validators import max_str_int
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine

# 数据库URL
ASYNC_DATABASE_URL = "mysql+aiomysql://root:MySQL901@localhost:3306/news_app?charset=utf8mb4"

# 创建异步引擎
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=True, # 可选：输出SQL日志
    pool_size=20, # 设置连接池中保持的持久连接数
    max_overflow=10 # 连接池溢出连接数
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












