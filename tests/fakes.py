"""测试用的 MySQL 替身：只实现项目里真正调用到的 AsyncSession 接口"""


class FakeResult:
    """模拟 SQLAlchemy Result"""

    def __init__(self, rows=None, scalar=None, one=None, rowcount=0):
        self._rows = rows if rows is not None else []
        self._scalar = scalar
        self._one = one
        self.rowcount = rowcount

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar

    def scalar_one_or_none(self):
        return self._one


class FakeSession:
    """记录 execute 调用次数，用来断言「有没有真的查库」"""

    def __init__(self, result=None):
        self.result = result or FakeResult()
        self.execute_calls = 0
        self.commits = 0
        self.added = []

    async def execute(self, stmt):
        self.execute_calls += 1
        return self.result

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def refresh(self, obj):
        pass

    def add(self, obj):
        self.added.append(obj)