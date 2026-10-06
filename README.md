# 今日头条后端 (Toutiao Backend)

基于 FastAPI 构建的异步新闻资讯后端服务，提供用户认证、新闻浏览、收藏、历史记录、AI 问答等完整功能。

## 技术栈

| 技术 | 版本/说明 |
|------|-----------|
| **Web 框架** | FastAPI |
| **异步服务器** | Uvicorn |
| **ORM** | SQLAlchemy 2.0 (Async) |
| **数据库** | MySQL 8.0+ |
| **异步驱动** | aiomysql |
| **缓存** | Redis |
| **数据验证** | Pydantic v2 |
| **密码加密** | bcrypt |
| **跨域处理** | FastAPI CORSMiddleware |
| **日志追踪** | logging + contextvars 请求 ID |
| **限流** | Redis 令牌桶（Lua 脚本保证原子性） |
| **容器化** | Docker + docker-compose（MySQL/Redis 一键起） |
| **测试** | pytest + pytest-asyncio（纯 mock，不依赖外部服务） |
| **CI** | GitHub Actions（Python 3.10/3.11/3.12 矩阵） |

## 项目结构

```
toutiao_backend/
├── main.py                     # FastAPI 应用入口（中间件、路由、日志初始化）
├── pytest.ini                  # pytest 配置（asyncio 自动模式）
├── requirements.txt            # 运行时依赖
├── requirements-dev.txt        # 开发/测试依赖
├── Dockerfile                  # 镜像定义（非 root 用户 + 健康检查）
├── docker-compose.yml          # MySQL + Redis + 应用一键编排
├── .dockerignore              # 排除 .env/.venv/__pycache__ 等
├── .github/workflows/test.yml  # CI：多版本矩阵跑 pytest
├── .env.example                # 环境变量示例（勿提交真实凭据）
├── cache/                      # 缓存业务层
│   └── news_cache.py          # 新闻相关缓存封装
├── config/                     # 配置文件
│   ├── db_conf.py             # 数据库配置 (MySQL)
│   └── cache_conf.py          # 缓存配置 (Redis)
├── models/                     # ORM 数据模型 (SQLAlchemy)
│   ├── base.py                # Pydantic 基础模型
│   ├── users.py               # 用户 & 用户令牌模型
│   ├── news.py                # 新闻分类 & 新闻模型
│   ├── favorite.py            # 收藏模型
│   └── history.py             # 浏览历史模型
├── schemas/                    # 请求/响应数据模型 (Pydantic)
│   ├── users.py               # 用户相关 DTO
│   ├── favorite.py            # 收藏相关 DTO
│   └── history.py             # 历史记录相关 DTO
├── crud/                       # 数据访问层 (业务逻辑)
│   ├── users.py               # 用户 CRUD
│   ├── news.py                # 新闻基础 CRUD
│   ├── news_cache.py          # 新闻带缓存的 CRUD
│   ├── favorite.py            # 收藏 CRUD
│   └── history.py             # 历史记录 CRUD
├── routers/                    # API 路由层
│   ├── users.py               # 用户相关接口
│   ├── news.py                # 新闻相关接口
│   ├── favorite.py            # 收藏相关接口
│   └── history.py             # 历史记录相关接口
├── ai/                         # AI 能力
│   ├── ai_chat.py             # 通用大模型 SSE 代理
│   ├── config.py              # AI 配置与阈值（集中管理，全部走环境变量）
│   ├── embeddings.py          # 向量化 + numpy 向量矩阵 + Redis 批量缓存
│   ├── retriever.py           # 混合检索 + RRF 融合 + 索引缓存 + 链路追踪
│   ├── prompts.py             # 带约束的 prompt 与拒答文案
│   ├── news_qa.py             # 新闻问答接口（SSE 流式 + 溯源）
│   ├── ingest/                # 离线索引管线
│   │   ├── validate.py        # 语料校验 + 内容指纹
│   │   ├── manifest.py        # 索引版本管理（兼容性判定）
│   │   ├── builder.py         # 构建/增量/断点续跑
│   │   ├── storage.py         # 落盘（.npy 向量 + JSON 元信息）
│   │   └── cli.py             # 命令行入口
│   └── eval/                  # 检索质量评估
│       ├── dataset.jsonl      # 55 条评估集（含难例与负例）
│       ├── runner.py          # 6 个指标 + 参数网格扫描
│       ├── fake_embedding.py  # 确定性假向量（离线验证融合机制）
│       ├── export_corpus.py   # 语料导出（供评估离线运行）
│       └── README.md          # 基线数据与已知局限
├── tests/                      # 单元测试 (pytest, 纯 mock)
│   ├── conftest.py            # 公共 fixture：假 Session / Redis 打桩 / 依赖覆盖
│   ├── fakes.py               # FakeResult、FakeSession
│   ├── test_auth_token.py     # Token 鉴权与 401 链路
│   ├── test_users_crud.py     # 密码哈希、Token 生命周期、部分字段更新
│   ├── test_news_cache.py     # 缓存命中/未命中、浏览量自增
│   ├── test_response.py       # 统一响应体与接口契约
│   ├── test_logging.py        # 请求 ID、日志初始化
│   ├── test_rate_limit.py     # 令牌桶突发/补充/并发原子性
│   ├── test_retrieval.py      # 分词、BM25、RRF 融合、prompt、防幻觉约束
│   ├── test_retrieval_perf.py # 检索性能回归哨兵 + 缓存行为
│   ├── test_retrieval_trace.py# 链路追踪写入（含 await 缺失的回归测试）
│   ├── test_ingest.py         # 语料校验、索引版本、增量更新、断点续跑
│   ├── test_ingest_integration.py # 离线索引与在线检索的衔接与校验
│   └── test_rag_quality.py    # 召回率与误召回率 CI 门禁
├── resourse/                   # 资源文件
│   └── database.sql           # 数据库初始化脚本（含 50+ 条种子数据）
└── utils/                      # 工具类
    ├── auth.py                # Token 认证依赖
    ├── security.py            # bcrypt 密码哈希
    ├── response.py            # 统一响应封装
    ├── logging_conf.py        # 日志配置 + contextvars 请求 ID
    ├── middleware.py          # 请求 ID 中间件 + 访问日志 + 耗时统计
    ├── rate_limit.py          # Redis 令牌桶限流
    ├── exception.py           # 异常处理器实现
    └── exception_handlers.py  # 全局异常注册
```

## 架构说明

本项目采用经典的分层架构设计，数据流自上而下单向依赖：

1. **Routers 层** (`routers/`)：定义 API 路由、请求参数校验、调用 CRUD 层，不写业务、不写 SQL
2. **CRUD 层** (`crud/`)：封装数据库操作、业务逻辑处理、缓存读写
3. **Models 层** (`models/`)：SQLAlchemy ORM 模型，映射数据库表
4. **Schemas 层** (`schemas/`)：Pydantic 模型，定义请求/响应结构
5. **Utils 层** (`utils/`)：通用工具（认证、安全、响应、异常、日志、限流）
6. **Cache 层** (`cache/`)：Redis 缓存操作封装
7. **Config 层** (`config/`)：数据库引擎和 Redis 连接配置

## 功能模块

### 1. 用户模块 (`/api/user`)

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/register` | 用户注册，返回 Token 和用户信息 | 否 |
| POST | `/login` | 用户登录，返回 Token 和用户信息 | 否 |
| GET | `/info` | 获取当前登录用户信息 | 是 |
| PUT | `/update` | 更新用户昵称/头像/性别/简介 | 是 |
| PUT | `/password` | 修改用户密码（需验证旧密码） | 是 |

**认证机制**：

- 注册/登录成功后生成 UUID Token，有效期 7 天
- Token 存储在 `user_token` 表中，关联用户 ID
- 后续请求通过 `Authorization: Bearer <token>` 请求头传递
- `utils/auth.py` 中的 `get_current_user` 作为 FastAPI 依赖注入自动校验
- 重新登录时覆盖旧 Token（一条用户对应一条有效 Token），且覆盖后**立即提交事务**，避免接口已返回新 Token 但数据库尚未落盘导致后续请求 401

**密码安全**：使用 bcrypt（rounds=12）加盐哈希，详见 `utils/security.py`

> 超过 72 字节的密码先做 sha256 摘要再 base64，而不是截断 —— 截断会让前 72 位相同的长密码得到相同哈希。
>
> 项目直接用原生 bcrypt 而非 passlib：passlib 最后一次发版是 2020 年，探测 bcrypt 后端时会用 73 字节探针密码，而 bcrypt 4.1+ 禁止超过 72 字节，两者组合会直接让注册接口 500。

### 2. 新闻模块 (`/api/news`)

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/categories` | 获取新闻分类列表 | 否 |
| GET | `/list` | 分页获取指定分类的新闻列表 | 否 |
| GET | `/detail` | 获取新闻详情（含浏览量+1和相关推荐） | 否 |

**缓存策略**（Redis）：

- **分类列表**：缓存 2 小时 (`news:categories`)
- **新闻列表**：按 `分类ID:页码:每页数量` 缓存 10 分钟 (`news_list:{分类ID}:{页码}:{数量}`)
- **新闻详情**：按 `新闻ID` 缓存 10 分钟 (`news_detail:{新闻ID}`)
- **相关推荐**：按 `新闻ID` 缓存 10 分钟 (`news_related:{新闻ID}`)
- **浏览量**：使用 Redis 原子递增 (`INCR`)，避免数据库并发写冲突，同时持久化到 MySQL

相关代码：

- 缓存封装：`cache/news_cache.py`
- 带缓存的 CRUD：`crud/news_cache.py`

### 3. 收藏模块 (`/api/favorite`)

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/check` | 检查指定新闻是否已收藏 | 是 |
| POST | `/add` | 添加新闻到收藏夹 | 是 |
| DELETE | `/remove` | 取消收藏指定新闻 | 是 |
| GET | `/list` | 分页获取收藏列表（按收藏时间倒序） | 是 |
| DELETE | `/clear` | 清空全部收藏 | 是 |

### 4. 浏览历史模块 (`/api/history`)

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/add` | 添加浏览历史（已存在则更新浏览时间） | 是 |
| GET | `/list` | 分页获取浏览历史（按浏览时间倒序） | 是 |
| DELETE | `/delete/{history_id}` | 删除单条浏览记录 | 是 |
| DELETE | `/clear` | 清空全部浏览历史 | 是 |

### 5. AI 问答模块 (`/api/ai`)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 通用大模型流式问答（透传 SSE） |
| POST | `/news-qa` | **新闻语料问答（RAG）**，SSE 流式，答案带新闻 ID 溯源 |
| POST | `/news-qa/sync` | 同上，一次性返回完整答案 |
| POST | `/news-qa/reload` | 强制刷新新闻语料索引缓存 |

`DASHSCOPE_API_KEY` 只从服务端环境变量读取，不下发到前端；未配置时相关接口返回 500并给出明确提示。

## 新闻问答（RAG）设计

数据流：

```
用户提问
  → 混合检索（BM25 词面 + 向量语义 + RRF 融合）召回相关新闻
  → 融合得分低于置信阈值？→ 直接拒答，不调用 LLM
  → 构造带约束的 prompt（新闻编号 + 来源）
  → SSE 流式返回，token 逐字透传
  → 溯源信息作为首个事件下发
```

实现见 `ai/retriever.py`、`ai/embeddings.py`、`ai/prompts.py`、`ai/news_qa.py`。

### 为什么必须先检索再问

模型不知道私有语料，而且它会编造看起来合理的新闻内容。新闻问答场景下这是最不能接受的错误类型。先把检索到的事实喂进去，再用 prompt 硬约束它只引用资料，答案才能溯源。

### 为什么要混合检索

两路的失败模式**互补**，这是混合检索的核心价值：

| | 强项 | 失败场景 |
|---|---|---|
| **BM25 词面** | 专有名词、数字、缩写。`GDP 增长5.2%` 只有词面能精确命中 | 同义表述零字面重叠 |
| **向量语义** | 跨越表述差异。「高铁延长运行时间」vs「动车组新增班次」 | 对专名不敏感，容易泛化 |

实测常见结论：单用向量会漏专名查询，单用 BM25 会漏同义表述，融合后召回率显著高于任一单路。

**为什么用 RRF 而不是直接相加分数**：BM25 分数无上界，向量余弦在[-1,1]，量纲不同不能相加。RRF 只依赖排名不看分数：`score(d) = Σ 1/(k + rank)`，k=60 是原论文经验值，作用是压制头部排名的过度影响。

### 防幻觉的三道约束

1. **中文分词要求至少命中一个多字词**。纯单字命中是噪声 —— 「量子计算机」和「人工智能」共享「计」「算」，加上「的/中/国」这类高频字，任何两条新闻之间都能凑出共同单字。只靠单字命中的结果直接丢弃，否则无关问题也会走上 LLM。

2. **单路命中的融合得分打 5 折**。两路都命中说明「从词面和语义两个角度都认为相关」，只有一路命中证据不足。
   单路最高分 `0.5/(60+1) ≈ 0.008`，双路最高分 `2/(60+1) ≈ 0.033`，置信阈值取 `0.02` 卡在两者之间。
   副作用要清楚：**向量服务不可用时系统进入保守模式，宁可拒答也不冒险让模型编。**

3. **Prompt 硬约束**：只能依据给定资料作答，资料没有就明说不知道，每条结论标注来源编号，多篇说法不一致时如实说明而不自行判断。

拒答时仍会把检索到的报道标题列给用户 —— 完全不给信息体验太差，而模型此时并没有编造的机会。

### 性能与缓存

检索有两处重开销，都随语料缓存而非每次重算：

| 环节 | 优化前 | 优化后 | 手段 |
|---|---|---|---|
| BM25 检索 | 74.3ms | **2.1ms** | 索引随语料缓存 |
| 向量检索 | 72.7ms | **0.26ms** | numpy 矩阵乘替代纯 Python 循环 |
| 语料向量读取 | 404 次 Redis 往返 | **1 次 MGET** | 批量读取 + 语料级向量缓存 |
| 完整检索（缓存命中） | ~1.6s | **0.1ms** | 上述叠加 |

语料指纹用 `(条数, 首尾 id, 首尾标题)` 计算，内容变化才失效。完整评估数据见 `ai/eval/README.md`。

### 检索链路追踪

每次检索记录完整链路，`AI_TRACE_ENABLED=true` 时按 `request_id` 落到 Redis：

```
timing  : {'bm25Ms': 2.07, 'vectorMs': 302.18, 'fuseMs': 0.09, 'totalMs': 304.35}
recall  : {'bm25Hits': 20, 'vectorHits': 0, 'vectorAvailable': False, 'degradedToBm25': True}
decision: {'topScore': 0.008197, 'threshold': 0.02, 'confident': False, 'willRefuse': True}
top1    : id=44 score=0.008197 bm25=1 vec=None single=True terms=['高铁']
```

能回答四类问题：慢在哪、**是否已降级为纯 BM25**、为什么拒答、命中了哪些词。
`degradedToBm25` 用来发现「以为在跑混合检索，实际向量路没生效」这种静默故障。

排查时按用户报的 `X-Request-ID` 取：
```bash
redis-cli -n 3 GET "ai:trace:<request_id>"
```

### 离线索引管线

显式构建流程，与检索时的懒加载并存。见 `ai/ingest/`：

```bash
python -m ai.ingest.cli validate                      # 只做语料校验，看脏数据
python -m ai.ingest.cli build --model text-embedding-v4   # 构建/增量更新
python -m ai.ingest.cli index-status                   # 查看索引落盘状态与兼容性
python -m ai.ingest.cli snapshots                      # 历史版本，可用于回滚
python -m ai.ingest.cli clear                          # 清空索引
```

**实测效果**（403 篇 × 1024 维）：

| 场景 | 向量化 | 复用 | API 调用 | 耗时 |
|---|---|---|---|---|
| 全量构建 | 403 | 0 | 403 | 2793ms |
| 内容不变 | 0 | 403 | **0** | — |
| 改 1 篇 | **1** | 402 | **1** | **428ms** |
| 换 embedding 模型 | 403 | 0 | 403 | 全量重建 |

四个设计要点：

- **索引版本（manifest）**：记录 embedding 模型、维度、语料指纹、文档数。
  换模型或语料变化时可立即判定失效 —— 维度不一致只会在检索时报 `index out of range`，
  很难联想到是模型换了。有了 manifest 还能回滚到历史版本。
- **增量更新**：按内容 hash 只处理变更文档。hash 只覆盖参与检索的字段
  （标题/描述/正文），改分类不触发重新向量化。
- **断点续跑**：进度存文件而非 Redis。中途失败重跑只处理未完成的部分。
  进度与本次待处理无交集时判定为陈旧并清除，避免把所有文档误判为已完成。
- **文件为准，Redis 为快路径**：Redis 不可用时索引产物依然可用。
  这条是被实测逼出来的 —— 最初把 manifest 只存 Redis，结果 Redis 一挂，
  增量判定全部失效，403 篇被重新向量化一次。

**数据层的坑**：真实语料里有 15 组共 33 篇**标题、描述、正文完全相同**的重复新闻
（如 id 22 与 116）。用 `hash -> 单个 id` 做映射会让后写入的覆盖先写入的，
那 33 篇永远进不了索引且没有任何报错。所以映射的值是 id 列表。

> **当前语料不做 chunking**：实测正文最长 192 字符、中位数 126，
> 全部是单段短新闻，一篇就是一个 chunk。分块策略（parent-child、重叠窗口）
> 在这个规模没有实际价值，等语料形态变化时再引入。

已建成评估体系（55 条评估集 + 6 个指标 + 参数扫描 + CI 门禁），当前基线：

| 指标 | @1 | @3 | @5 |
|---|---|---|---|
| Recall@K | 87.8% | 93.9% | 93.9% |
| MRR@K | 87.8% | 90.5% | 90.5% |
| nDCG@K | 87.8% | 91.4% | 91.4% |

误召回率 0%。用法与已知局限见 `ai/eval/README.md` —— 那里记录了两个必须知道的限制：
阈值尚未标定、评估集的 gold 是按标题反推而非人工标注。

### 已知限制

- 置信阈值是按「单路/双路最高分的量级」手工校准的，**没有用标注集做量化评估**。要更严谨应该建一份QA 评测集，用 Recall@K / MRR 扫参数。
- 向量服务不可用时会偏保守（见上文第 2 条）。

## 数据库设计

数据库初始化脚本见 `resourse/database.sql`。

> 脚本第一行的 `SET NAMES utf8mb4` 不能删。MySQL 客户端的默认字符集不一定是 utf8mb4（Docker 官方镜像初始化时就会出问题），少了这一行，中文会先按 latin1 读入再按 utf8mb4 存一次，变成双重编码的乱码，且乱码会直接写进库里。

### 核心数据表

1. **`user` 用户表**
   - 字段：id, username(唯一), password(bcrypt 哈希), nickname, avatar, gender, bio, phone(唯一), created_at, updated_at

2. **`user_token` 用户令牌表**
   - 字段：id, user_id(FK→user), token(唯一), expires_at, created_at

3. **`news_category` 新闻分类表**
   - 字段：id, name(唯一), sort_order, created_at, updated_at
   - 默认分类：头条、社会、国内、国际、娱乐、体育、科技、财经

4. **`news` 新闻表**
   - 字段：id, title, description, content, image, author, category_id(FK→news_category), views, publish_time, created_at, updated_at
   - 索引：category_id、publish_time

5. **`related_news` 相关新闻关联表**（预留推荐系统用）
   - 字段：id, news_id, related_news_id, created_at

6. **`favorite` 收藏表**
   - 字段：id, user_id(FK→user), news_id(FK→news), created_at
   - 唯一约束：(user_id, news_id)

7. **`history` 浏览历史表**
   - 字段：id, user_id(FK→user), news_id(FK→news), view_time
   - 唯一约束：(user_id, news_id)
   - 索引：view_time（倒序，支持历史列表排序）

8. **`ai_chat` AI聊天记录表**（预留）
   - 字段：id, user_id(FK→user), message, response, created_at

## 统一响应格式

所有接口使用统一的 JSON 响应格式：

```json
{
  "code": 200,
  "message": "success",
  "data": {}
}
```

- `code`: 状态码（200 成功，4xx 客户端错误，5xx 服务端错误）
- `message`: 提示信息
- `data`: 响应数据（可能为对象、数组、null）

异常响应额外带 `request_id` 字段，方便用户报障时定位服务端日志。

实现见 `utils/response.py`

## 全局异常处理

项目注册了 4 层异常处理器，具体见 `utils/exception_handlers.py`：

1. **HTTPException**（`fastapi.HTTPException`）：处理 FastAPI 抛出的 HTTP 异常（401、404 等）
2. **IntegrityError**（`sqlalchemy.exc.IntegrityError`）：处理 MySQL 完整性约束冲突（重复用户名、外键不存在等）
3. **SQLAlchemyError**（`sqlalchemy.exc.SQLAlchemyError`）：处理所有 SQLAlchemy 数据库错误
4. **Exception**：兜底捕获所有未预期异常

> 注册顺序按「子类在前、父类在后」，否则 `IntegrityError` 会被 `SQLAlchemyError` 抢先接管，友好提示就丢失了。

`DEBUG_MODE` 从环境变量读取且**默认为 false**。开启后会在 `data` 字段返回详细错误信息和堆栈追踪，生产环境必须关闭 —— 堆栈只应该出现在服务端日志里。

## 日志与请求追踪

每个请求都会生成唯一 ID（优先复用上游传入的 `X-Request-ID`），通过响应头返回给前端，并注入到该请求的全部日志里：

```
2026-10-05 17:01:28 | INFO | req=a6842e44 | utils.middleware | GET /api/news/categories -> 200 耗时 13.4ms
```

这样用户报障时只要提供 `X-Request-ID`，就能在日志里 grep 出整条链路（含耗时和异常栈）。

实现要点见 `utils/logging_conf.py` 和 `utils/middleware.py`。

> **为什么用 `contextvars` 而不是 `threading.local`？**
> asyncio 是单线程协作式调度，多个请求的协程交替跑在同一个线程里。`threading.local` 只有一个副本，分不清请求；`contextvars` 是 per-Task 存储，asyncio 每创建一个 Task 就复制一份上下文，所以各请求互不干扰，又能在任意深的调用栈里通过 `get_request_id()` 取到。

## 接口限流

基于 Redis 的**令牌桶（Token Bucket）**，按「客户端 IP + 请求路径」分桶，见 `utils/rate_limit.py`。

```
ratelimit:{路径}:{IP}    -> HASH { tokens: 当前令牌数, ts: 上次更新时间 }
```

桶初始装满 `capacity` 个令牌；每过 1 秒按 `rate` 个/秒匀速补充，上限封顶 `capacity`。请求消耗 1 个令牌，不够就返回 429：

```
HTTP/1.1 429 Too Many Requests
Retry-After: 1
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 0
```

实测（capacity=5、rate=2/s）：突发 5 次全放行，第 6 次 429；等 1.5 秒补回 3 个令牌后，又能放行 3 次。

几个设计取舍：

| 取舍 | 说明 |
|---|---|
| **令牌桶 vs 固定窗口** | 固定窗口把时间切成互不相干的段，窗口交界处可以打满两次，1 秒内放行 2N 个请求。令牌桶按速率匀速放行 + 允许有限突发，同时约束长期平均速率和瞬时上限，是 AWS/Stripe 的做法。 |
| **Lua 脚本 vs 多条命令** | 令牌桶是「读 tokens → 按流逝时间补充 → 扣减 → 写回」四步。拆成多条命令时，两个并发请求可能读到同一个 `tokens=0` 然后都通过校验，导致放行量翻倍。Redis 的 Lua 脚本在执行期间原子，这一步由服务端保证，不需要分布式锁。测试里用 50 并发打一个 capacity=20 的桶，断言恰好放行 20 个。 |
| **Redis vs 进程内计数** | 进程内 Map 在多 worker 部署下每个进程各算各的，实际阈值会放大 N 倍。Redis 是共享存储，配合 Lua 天然全集群统一计数。 |
| **时间戳由客户端传入** | 避免 Redis 主从切换时 `TIME` 返回值跳变导致计数异常。代价是依赖各机器时钟大致同步（NTP）。脚本里对 `now < ts`（时钟回拨）按 0 处理，避免一次时间跳变就绕过限流。 |
| **返回值用 `tostring`** | Redis 会把 Lua 返回值转成整数，直接返回浮点令牌数会被截断，表现为实际限流比配置的更严。 |
| **Redis 故障时 fail-open** | 限流是保护措施而非业务逻辑，挡不住流量时让请求继续走，比整个服务不可用更合理。 |

参数用 `RATE_LIMIT_CAPACITY`（默认 100，突发上限）和 `RATE_LIMIT_RATE`（默认 100/秒，长期速率）配置，`RATE_LIMIT_CAPACITY=0` 可关闭。`/docs`、`/redoc`、`/openapi.json` 不限流。

> `X-Forwarded-For` 只在**可信反向代理**后面才应该信任，否则客户端可以伪造该头绕过限流。生产部署需保证前置代理清洗这些头。

## 持续集成

`.github/workflows/test.yml` 在 push 和 PR 时自动运行：

- Python **3.10 / 3.11 / 3.12** 三版本矩阵（同时验证版本兼容性）
- 语法编译检查 + `pytest -v`
- 主分支额外跑一次冒烟：导入应用并校验 22 条路由

因为测试不依赖外部服务，CI 不需要配 MySQL/Redis 容器，配置极简、跑得快。

## 环境要求

- Python 3.10+
- MySQL 8.0+
- Redis 6.0+

> 不装本地 MySQL/Redis 也行 —— 用 Docker 一键起完整环境。

## 快速开始（Docker，推荐）

```bash
docker compose up -d --build
curl http://localhost:3001/api/news/categories
```

一条命令拉起三个服务：**MySQL 8 + Redis + 应用**。

- 首次启动自动执行 `resourse/database.sql`，建表并导入 50+ 条新闻种子数据，无需手动导 SQL
- 应用等 MySQL/Redis **healthcheck 通过**后才启动（`depends_on: condition: service_healthy`），不会出现「容器起来了但数据库还没初始化」的启动竞态
- 镜像以**非 root 用户**运行，内置 `HEALTHCHECK`
- 宿主机端口可配：本机已装 MySQL/Redis 时用 `MYSQL_PORT=3308 REDIS_PORT=6381 docker compose up -d` 避开冲突

常用命令：

```bash
docker compose logs -f app      # 看应用日志（含 request_id）
docker compose ps               # 看健康状态
docker compose down             # 停止
docker compose down -v          # 停止并删除数据卷（会清库）
```

## 本地开发

### 1. 安装依赖

```bash
pip install -r requirements.txt        # 运行时依赖
pip install -r requirements-dev.txt    # 开发/测试依赖
```

### 2. 配置环境变量

所有配置都从环境变量读取，**代码里不硬编码任何凭据**。可复制 `.env.example` 作为参考：

```bash
# 数据库
export DB_USER=root
export DB_PASSWORD=你的密码
export DB_HOST=localhost
export DB_PORT=3306
export DB_NAME=news_app
export DB_CHARSET=utf8mb4
export DB_ECHO=false          # 是否打印 SQL 日志
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=10

# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=3

# 日志与错误信息
export LOG_LEVEL=INFO        # DEBUG / INFO / WARNING / ERROR
export LOG_FILE=             # 留空只输出到控制台；填路径则同时写文件（自动轮转）
export DEBUG_MODE=false      # true 会把异常堆栈返回给前端，仅本地调试用

# 服务
export HOST=0.0.0.0
export PORT=3001
export CORS_ORIGINS=*        # 生产环境填具体域名，逗号分隔

# 限流（令牌桶）
export RATE_LIMIT_CAPACITY=100   # 桶容量，允许的瞬时突发上限，设 0 关闭
export RATE_LIMIT_RATE=100       # 每秒补充令牌数，决定长期平均速率

# AI 大模型（可选，不配置则 /api/ai/* 返回 500）
export DASHSCOPE_API_KEY=sk-xxxx
export DASHSCOPE_MODEL=qwen-plus
export DASHSCOPE_EMBED_MODEL=text-embedding-v4

# 新闻问答检索参数
export AI_TOP_K_BM25=20          # BM25 每路召回条数
export AI_TOP_K_VECTOR=20        # 向量每路召回条数
export AI_TOP_K_FINAL=5# 喂给 LLM 的新闻条数
export AI_RRF_K=60                # RRF 平滑常数
export AI_SINGLE_PATH_WEIGHT=0.5  # 单路命中惩罚系数
export AI_MIN_FUSION_SCORE=0.02   # 低于此分判定为无相关报道，直接拒答
export AI_MAX_DOC_CHARS=400       # 每条新闻最多取多少字进上下文
export AI_TRACE_ENABLED=false     # 开启后按 request_id 把检索链路落到 Redis
```

### 6. 运行检索质量评估

```bash
python -m ai.eval.export_corpus                # 导出语料（改了 database.sql 需重跑）
python -m ai.eval.runner --baseline --k 1,3,5,8
python -m ai.eval.runner --sweep --top-k 5     # 参数网格扫描，输出权衡表
python -m ai.eval.runner --baseline --vector-real   # 需有效 API Key
```

所有变量都有默认值，未设置时回落到本地开发配置。

### 3. 初始化数据库

```bash
mysql -u root -p < resourse/database.sql
```

### 4. 启动服务

```bash
python main.py
```

或使用 uvicorn 直接启动：

```bash
uvicorn main:app --host 0.0.0.0 --port 3001 --reload
```

服务启动后访问：

- API 服务地址：http://localhost:3001
- FastAPI 自动文档 (Swagger UI)：http://localhost:3001/docs
- 备用文档 (ReDoc)：http://localhost:3001/redoc

### 5. 运行测试

测试采用纯 mock 策略，**不依赖 MySQL 和 Redis**，本地/CI 都能直接跑：

```bash
pytest -q
```

覆盖内容：

- **Token 鉴权**：token 有效 / 已过期 / 不存在，鉴权失败必须返回 401（而非 500）
- **密码安全**：bcrypt 加盐、同密码不同哈希、超 72 字节密码、脏哈希容错
- **缓存策略**：分类与列表缓存的命中（不查库）、未命中（回源并回写）、空结果不写缓存、缓存 key 页码推导
- **浏览量**：Redis `INCR` 与 MySQL `UPDATE` 各调用一次
- **序列化**：缓存写入前 `datetime` 已被转成 ISO 字符串，避免 `json.dumps` 抛 `TypeError`
- **日志追踪**：每个响应带 `X-Request-ID`、上游传入的 ID 会被复用、`contextvars` 请求结束后正确还原
- **限流**：令牌桶突发容量与补充速率、50 并发不超发的原子性验证、按 IP+路径分桶、时钟回拨防护、Redis 故障 fail-open
- **检索**：中文分词的单字/二元组、BM25 专名精确匹配、单字噪声过滤、RRF 融合数学、单路命中惩罚、余弦零向量保护
- **性能回归哨兵**：BM25 < 25ms、向量 < 5ms、缓存必须命中且不得无界增长
- **链路追踪**：trace 落 Redis、`request_id` 关联、TTL 正确、降级模式被标记、写失败不影响主流程
- **离线索引**：语料脏数据拦截、索引兼容性判定、增量只处理变更文档、断点续跑、重复文档不丢失
- **防幻觉**：SSE 溯源事件可 JSON 序列化（datetime 边界处理）、拒答文案包含已检索到的报道、system prompt 约束存在性
- **接口契约**：所有接口返回 `code`/`message`/`data`，响应体中不再出现 `msg`

## 关键代码示例

### 数据库会话管理 (异步)

`config/db_conf.py` 中使用 FastAPI 依赖注入模式管理异步数据库会话，自动提交/回滚/关闭：

```python
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()
```

### Token 认证依赖

`utils/auth.py` 中作为 FastAPI `Depends` 使用，自动从 `Authorization` 头提取 Token 并查询用户：

```python
@router.get("/info")
async def get_user_info(user: User = Depends(get_current_user)):
    return success_response(data=user)
```

### 联表查询 (收藏列表)

`crud/favorite.py` 中使用 SQLAlchemy `join` 关联 News 和 Favorite 表：

```python
query = (select(News, Favorite.created_at.label("favorite_time"), Favorite.id.label("favorite_id"))
         .join(Favorite, Favorite.news_id == News.id)
         .where(Favorite.user_id == user_id)
         .order_by(Favorite.created_at.desc())
         .offset(offset)
         .limit(page_size))
```

### 跨域配置

`main.py` 中配置了全局 CORS 中间件，来源可通过 `CORS_ORIGINS` 配置，开发环境允许所有来源：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)
```

> 生产环境请务必将 `CORS_ORIGINS` 设置为具体前端域名。