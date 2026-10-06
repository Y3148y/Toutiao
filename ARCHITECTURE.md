# 项目架构与核心概念笔记

本文档梳理本项目（FastAPI + SQLAlchemy + Redis）的核心概念，帮助理解代码逻辑与分层架构。

---

## 1. `add()` 和 `commit()` 的使用时机

SQLAlchemy 的 session 是「工作单元」模式，`add` 和 `commit` 分工不同：

| 操作 | 是否需要 add/commit |
|------|---------------------|
| 查询（select） | 都不需要，`await db.execute(stmt)` 即可 |
| 新增（insert） | `add` → `commit` |
| 修改（update） | `commit`（两种写法见下） |
| 删除（delete） | `commit` |

- **`add()`**：把新建的 ORM 对象挂到 session 的「待处理队列」，标记为 INSERT，但不执行 SQL。
- **`commit()`**：把 session 里所有 pending 变更真正 flush 生成 SQL 并提交事务。

### 新增示例（create_user）

```python
async def create_user(db, user_data):
    hashed_password = security.get_hash_password(user_data.password)
    user = User(username=user_data.username, password=hashed_password)
    db.add(user)          # 1. 加入待插入队列
    await db.commit()     # 2. 执行 INSERT 并提交
    await db.refresh(user)# 3. 回读数据库（拿到自增 id 等）
    return user
```

### 修改的两种写法

- 写法 A：用 ORM 对象改属性（`db.add` 确保对象仍被 session 跟踪）

```python
user.password = new_hash_password
db.add(user)
await db.commit()
```

- 写法 B：用 SQLAlchemy 的 `update()` 语句

```python
stmt = update(News).where(News.id == news_id).values(views=News.views + 1)
await db.execute(stmt)
await db.commit()
```

### 为什么查询不需要 commit

session 有 `expire_on_commit=False` 且查询是只读的，`execute(select(...))` 只发起查询拿到结果，不产生事务变更。

### `get_db` 的自动 commit

`yield session` 结束后自动 `commit`，异常时 `rollback`，最后 `close`。这是兜底，但业务代码里通常仍显式 `commit` 更可控。

---

## 2. 为什么要用 `alias`（字段别名）

核心原因：**Python 变量名和数据库列都是 snake_case，但前端 JSON 约定是 camelCase**。

```python
category_id: int = Field(alias="categoryId")        # 对外叫 categoryId
publish_time: Optional[datetime] = Field(None, alias="publishedTime")
```

配套配置（缺了会出问题）：

```python
model_config = ConfigDict(
    from_attributes=True,   # 允许从 ORM 对象读属性（model_validate(orm)）
    populate_by_name=True   # 允许用原始字段名(snake_case)填充，而不只能用 alias
)
```

- **`alias`**：序列化给前端时，把 `category_id` 输出成 `categoryId`。
- **`populate_by_name=True`**：若只有 `alias` 没有它，`model_validate(orm)` 时会拿 alias（`categoryId`）去 ORM 对象找属性，但 ORM 对象属性是 `category_id`，找不到 → 校验失败/字段丢失。
- **`by_alias=False`**：`model_dump(mode="json", by_alias=False)` 输出 snake_case 字段名（后端内部用），默认 `by_alias=True` 输出 camelCase（给前端）。

一句话：**alias 负责"对外改名字"，populate_by_name 负责"对内两种名字都认"，by_alias=False 负责"内部用回原名"**。

---

## 3. json / 字符串 / dict、list / ORM 的转换链路

### 写入缓存（ORM → JSON 字符串）

```
ORM对象(NEWS) → Pydantic(model_validate) → dict(model_dump) → json字符串(json.dumps) → Redis
```

```python
news_data = [NewsItemBase.model_validate(item).model_dump(mode="json", by_alias=False)
             for item in cached_news_lists]
await set_cache_news_list(category_id, page, limit, news_data)
```

分工：

- **`model_validate(orm)`**：ORM 对象 → Pydantic 模型（配合 `from_attributes=True` 读属性）。
- **`model_dump(mode="json")`**：Pydantic → 纯 dict，`mode="json"` 把 `datetime` 转 ISO 字符串。
- **`jsonable_encoder(obj)`**：FastAPI 万能转换器，把 ORM/Pydantic/datetime 递归转成可 JSON 类型。

`json.dumps` 在底层缓存实现里完成（dict/list → json 字符串）。

### 读取缓存（JSON 字符串 → ORM）

```
Redis字符串 → dict(json.loads) → ORM对象(News(**dict))  或  直接返回 dict
```

- `get_json_cache` 负责 `json.loads`。
- news_list 读缓存用 `[News(**item) for item in cached]` 把 dict 变回 ORM。

### ⚠️ 关键坑

**datetime 不能直接 `json.dumps`**。必须先用 `mode="json"` 或 `jsonable_encoder` 转字符串，否则 `json.dumps` 抛 `TypeError: Object of type datetime is not JSON serializable`，又被 except 吞掉（静默失败）→ 缓存永远写不进去。

---

## 4. Pydantic / BaseModel / model 类 / schema 类的区别

| 概念 | 基类 | 归属目录 | 职责 |
|------|------|----------|------|
| SQLAlchemy model 类（News, User...） | `DeclarativeBase` | models/ | 映射数据库表，字段是 DB 列 |
| Pydantic schema 类（UserRequest...） | `BaseModel` | schemas/ | 定义 API 请求/响应结构（DTO） |

- **Pydantic** = 库名；**BaseModel** = 它的基类；**schema 类** = 继承 BaseModel 的具体 DTO。
- **注意**：项目中 `models/base.py` 的 `NewsItemBase` / `NewsItemFull` 其实也是 Pydantic（继承 BaseModel），不是 SQLAlchemy，是「新闻响应的序列化辅助模型」。

### 两者的桥接（关键）

```python
model_config = ConfigDict(from_attributes=True)
```

有了它，`UserInfoResponse.model_validate(user_orm)` 能直接从 SQLAlchemy 对象读属性并映射成 JSON。

### 分工总结

- `models/` 的 ORM 类 → 管**数据库怎么存**
- `schemas/` 的 DTO 类 → 管**接口怎么传**
- 通过 `from_attributes=True` + `model_validate` 完成 ORM → DTO 转换

---

## 5. 分层架构

项目是典型的 7 层架构，数据流自上而下单向依赖：

```
请求进来
  ↓
routers/     路由层：参数解析、依赖注入、调用下层、组装响应，不写业务
  ↓
crud/        数据访问层：写 SQL、缓存读写、业务逻辑（核心）
  ↓          ↘
models/       cache/           config/
ORM(DB表映射)  Redis缓存封装     连接配置(MySQL/Redis)
  ↑
schemas/     Pydantic DTO（贯穿进/出两层）
utils/       认证、密码、响应、异常
```

各层职责：

1. **routers**：定义 URL 路径、HTTP 方法、参数类型（Query/Path/Body）、依赖注入（Depends）。不含 SQL。
2. **crud**：封装 select/update/delete，决定「先查缓存还是查库」。
3. **models**：SQLAlchemy ORM，一张表对应一个类。
4. **schemas**：Pydantic DTO，定义接口入参/出参和校验规则。
5. **cache**：把 Redis get/set 封装成业务语义函数，统一 key 生成。
6. **config**：数据库引擎、session 工厂、Redis 连接、底层缓存原语。
7. **utils**：横切关注点——token 认证、bcrypt 密码、统一响应、全局异常。

### 一个完整数据流（请求新闻详情）

```
GET /api/news/detail?id=1
  → routers/news.py: read_news_detail（解析 id，注入 db）
  → crud/news_cache.py: get_news_detail（先查 Redis）
      → 命中：直接返回
      → 未命中：select(News) → 写 Redis
  → crud/news_cache.py: increase_news_news（Redis INCR + DB UPDATE views）
  → crud/news_cache.py: get_related_news（相关推荐，同样走缓存）
  → 组装 dict → FastAPI 自动 jsonable_encoder → JSON 响应
```