# FastAPI 依賴注入設計模式

## 簡介

依賴注入是 FastAPI 的核心特性之一，通過 `Depends` 函數實現。它能幫助我們構建更加模組化、可測試且易於維護的應用程式。本文將介紹 FastAPI 中常用的依賴注入設計模式，並提供實用的程式碼範例。

## 為什麼使用依賴注入？

依賴注入為我們的 FastAPI 應用提供以下優勢：

- **解耦**：將依賴的創建與使用分離，降低模組間的耦合度
- **可測試性**：輕鬆替換依賴項以進行單元測試
- **重用性**：在多個路由中重複使用相同的依賴
- **可維護性**：更清晰的代碼結構和責任分離

## 常用設計模式

### 1. 基礎依賴注入模式

最簡單的依賴注入形式，直接在路由函數中使用 `Depends`。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

def get_query_params(q: str = None):
    return {"q": q}

@app.get("/items/")
def read_items(params: dict = Depends(get_query_params)):
    return {"params": params}
```

**適用場景**：處理簡單的依賴關係，如查詢參數處理、簡單的配置讀取等。

### 2. 單例模式

確保某個依賴在整個應用中只有一個實例，適用於共享資源或配置。

```python
from fastapi import FastAPI, Depends
from functools import lru_cache

app = FastAPI()

@lru_cache()
def get_settings():
    # 在實際應用中，這可能會從環境變數或配置文件讀取
    return {"app_name": "Awesome API", "version": "1.0.0"}

@app.get("/info")
def read_info(settings: dict = Depends(get_settings)):
    return settings
```

**適用場景**：應用配置、資料庫連接池、共享客戶端等需要全局共享的資源。

### 3. 工廠模式

動態創建依賴實例，可以根據不同條件返回不同的實例。

```python
from fastapi import FastAPI, Depends, Header

app = FastAPI()

def get_db_connection(x_environment: str = Header(None)):
    # 根據請求頭中的環境變數選擇不同的資料庫連接
    if x_environment == "testing":
        return {"connection": "test_db_connection"}
    else:
        return {"connection": "production_db_connection"}

@app.get("/data")
def read_data(db: dict = Depends(get_db_connection)):
    return {"database": db["connection"]}
```

**適用場景**：需要根據請求上下文動態選擇依賴實現的場景。

### 4. 鏈式依賴模式

一個依賴可以依賴於其他依賴，形成依賴鏈。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

def get_token(authorization: str = Header(None)):
    return authorization

def get_current_user(token: str = Depends(get_token)):
    # 在實際應用中，這裡會驗證 token 並返回用戶
    return {"user": "john_doe", "token": token}

@app.get("/users/me")
def read_user_me(current_user: dict = Depends(get_current_user)):
    return current_user
```

**適用場景**：複雜的授權流程、多層業務邏輯處理等。

### 5. 裝飾器模式

使用依賴注入來裝飾路由函數，增加額外的功能。

```python
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

def verify_admin(x_role: str = Header(None)):
    if x_role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    return True

@app.get("/admin", dependencies=[Depends(verify_admin)])
def admin_route():
    return {"message": "Welcome, admin!"}
```

**適用場景**：權限驗證、請求日誌記錄、性能監控等橫切關注點。

### 6. 類別依賴模式

使用類別而不是函數來定義依賴，提供更好的封裝和組織。

```python
from fastapi import FastAPI, Depends

app = FastAPI()

class DatabaseClient:
    def __init__(self, db_name: str = "default"):
        self.db_name = db_name
        # 在實際應用中，這裡會初始化資料庫連接
        
    def get_items(self):
        # 模擬從資料庫獲取數據
        return [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

@app.get("/items")
def read_items(db: DatabaseClient = Depends(DatabaseClient)):
    return db.get_items()
```

**適用場景**：複雜的依賴邏輯，需要封裝狀態和行為的場景。

## 最佳實踐

1. **保持依賴函數簡單**：每個依賴函數應該專注於單一職責。

2. **適當使用緩存**：對於計算成本高但不常變化的依賴，使用 `@lru_cache` 裝飾器。

3. **考慮依賴的生命週期**：了解依賴在請求中的生命週期，避免不必要的資源消耗。

4. **使用類型提示**：充分利用 Python 的類型提示功能，提高代碼的可讀性和 IDE 支持。

5. **分層組織依賴**：將相關的依賴組織在一起，形成清晰的依賴層次結構。

## 結論

FastAPI 的依賴注入系統非常靈活且強大，通過合理運用這些設計模式，我們可以構建出結構清晰、易於測試且高度可維護的 API 應用。依賴注入不僅是一種技術實現，更是一種設計思想，幫助我們寫出更好的代碼。
