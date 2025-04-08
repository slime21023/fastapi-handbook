# FastAPI 依賴注入的性能優化

## 簡介

依賴注入為 FastAPI 應用提供了強大的模組化和可測試性，但如果使用不當，可能會導致性能問題。本文將探討 FastAPI 依賴注入的性能考量，並提供優化策略和最佳實踐，幫助您構建既靈活又高效的 API。

## 依賴注入的性能影響

在討論優化之前，讓我們先了解依賴注入可能對性能產生的影響：

1. **依賴解析開銷**：每次請求都需要解析和執行依賴函數
2. **多餘的計算**：未經優化的依賴可能重複執行相同的計算
3. **資源管理**：依賴中的資源（如資料庫連接）需要妥善管理
4. **依賴鏈深度**：過長的依賴鏈可能導致性能下降

## 性能優化策略

### 1. 使用緩存減少重複計算

對於計算成本高但結果不常變化的依賴，可以使用 `functools.lru_cache` 進行緩存。

```python
from functools import lru_cache
from fastapi import FastAPI, Depends

app = FastAPI()

# 未優化版本
def get_settings():
    # 假設這是一個成本較高的操作，如讀取配置文件
    return {"app_name": "MyApp", "version": "1.0"}

# 優化版本
@lru_cache()
def get_cached_settings():
    # 相同的操作，但結果會被緩存
    return {"app_name": "MyApp", "version": "1.0"}

@app.get("/config")
def read_config(settings=Depends(get_cached_settings)):
    return settings
```

**效果**：第一次調用後，後續請求將直接使用緩存結果，避免重複計算。

**適用場景**：配置讀取、靜態資源加載等不常變化的依賴。

### 2. 優化依賴的作用域

FastAPI 允許我們定義依賴的作用域，這對於資源管理非常重要。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

# 請求級別依賴（默認）
def get_request_db():
    db = Database()
    try:
        yield db
    finally:
        db.close()  # 請求結束後關閉連接

# 應用級別依賴
@app.on_event("startup")
def create_app_db():
    app.db = Database()

@app.on_event("shutdown")
def close_app_db():
    app.db.close()

def get_app_db():
    return app.db

@app.get("/items")
def read_items(db=Depends(get_app_db)):
    return db.get_items()
```

**效果**：根據需求選擇適當的作用域，避免不必要的資源創建和銷毀。

**適用場景**：資料庫連接池、API 客戶端等需要管理生命週期的資源。

### 3. 減少依賴鏈的深度

過長的依賴鏈會增加解析時間和複雜性，應適當優化。

```python
# 過長的依賴鏈
def get_settings():
    return {"db_url": "postgresql://user:pass@localhost/db"}

def get_db(settings=Depends(get_settings)):
    return Database(settings["db_url"])

def get_user_repo(db=Depends(get_db)):
    return UserRepository(db)

def get_auth_service(repo=Depends(get_user_repo)):
    return AuthService(repo)

@app.get("/users/me")
def read_current_user(auth=Depends(get_auth_service), token: str = Header(None)):
    return auth.get_current_user(token)

# 優化後的依賴鏈
def get_auth_service():
    settings = get_settings()
    db = Database(settings["db_url"])
    repo = UserRepository(db)
    return AuthService(repo)

@app.get("/users/me")
def read_current_user(auth=Depends(get_auth_service), token: str = Header(None)):
    return auth.get_current_user(token)
```

**效果**：減少依賴解析的層級，降低性能開銷。

**適用場景**：複雜的業務邏輯，過長的依賴鏈。

### 4. 使用異步依賴

對於 I/O 密集型操作，使用異步依賴可以顯著提高性能。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

# 同步版本
def get_data():
    # 假設這是一個 I/O 密集型操作
    import time
    time.sleep(1)  # 模擬 I/O 等待
    return {"data": "example"}

# 異步版本
async def get_data_async():
    # 使用異步操作替代阻塞調用
    import asyncio
    await asyncio.sleep(1)  # 非阻塞等待
    return {"data": "example"}

@app.get("/data-async")
async def read_data_async(data=Depends(get_data_async)):
    return data
```

**效果**：在高並發場景下，異步依賴可以更有效地利用系統資源。

**適用場景**：資料庫查詢、API 調用等 I/O 密集型操作。

### 5. 使用類別依賴的懶加載

類別依賴可以實現懶加載模式，僅在需要時初始化資源。

```python
class LazyResource:
    def __init__(self):
        self._resource = None
    
    @property
    def resource(self):
        if self._resource is None:
            # 僅在首次訪問時初始化
            self._resource = ExpensiveResource()
        return self._resource

class ServiceWithLazyLoading:
    def __init__(self):
        self.lazy_resource = LazyResource()
    
    def get_data(self):
        # 只有在調用此方法時才會初始化資源
        return self.lazy_resource.resource.get_data()

@app.get("/lazy-data")
def read_lazy_data(service: ServiceWithLazyLoading = Depends(ServiceWithLazyLoading)):
    return service.get_data()
```

**效果**：避免在每個請求中都初始化所有資源，只加載實際需要的資源。

**適用場景**：包含多個可能不會全部使用的重量級資源的依賴。

## 性能監控與分析

優化性能的第一步是了解當前的性能瓶頸。以下是一些監控和分析 FastAPI 應用性能的方法：

### 1. 使用中間件測量依賴解析時間

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 2. 使用分析工具

- **cProfile**：Python 的內置分析器
- **pyinstrument**：更現代的 Python 分析器，提供可視化報告
- **OpenTelemetry**：用於分布式追蹤的開源框架

```python
# 使用 cProfile 分析依賴性能
import cProfile

def profile_dependency():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 執行依賴函數
    result = expensive_dependency()
    
    profiler.disable()
    profiler.print_stats(sort='cumtime')
    
    return result
```

## 最佳實踐

1. **優先考慮可讀性和可維護性**：除非有明確的性能問題，否則不要過早優化。

2. **適當使用緩存**：對於計算成本高但結果不常變化的依賴，使用緩存。

3. **選擇正確的作用域**：根據資源的性質選擇適當的生命週期管理策略。

4. **監控關鍵依賴**：持續監控關鍵依賴的性能，及時發現問題。

5. **負載測試**：在真實負載下測試應用，確保依賴注入系統能夠處理高並發請求。

6. **使用連接池**：對於資料庫等資源，使用連接池而不是每次請求創建新連接。

```python
from databases import Database
from fastapi import FastAPI, Depends

app = FastAPI()

# 應用啟動時創建連接池
@app.on_event("startup")
async def startup():
    app.db = Database("postgresql://user:pass@localhost/db")
    await app.db.connect()

@app.on_event("shutdown")
async def shutdown():
    await app.db.disconnect()

# 依賴函數返回連接池
async def get_db():
    return app.db

@app.get("/users/{user_id}")
async def read_user(user_id: int, db=Depends(get_db)):
    query = "SELECT * FROM users WHERE id = :user_id"
    return await db.fetch_one(query=query, values={"user_id": user_id})
```

## 結論

FastAPI 的依賴注入系統提供了強大的功能，但需要謹慎使用以避免性能問題。通過適當的緩存策略、資源管理、依賴結構優化和異步處理，我們可以在保持代碼可維護性的同時，確保應用具有良好的性能。

記住，性能優化應該是一個持續的過程，基於實際測量和監控數據進行，而不是基於假設。在大多數情況下，清晰、可維護的代碼結構比過早的性能優化更重要。