# FastAPI 依賴注入實戰：身份驗證模式

## 簡介

身份驗證是 Web API 中不可或缺的一部分，而 FastAPI 的依賴注入系統為實現靈活且安全的身份驗證提供了絕佳的基礎。本文將探討如何利用依賴注入實現各種身份驗證策略，從基本的 API 密鑰驗證到更複雜的 OAuth2 和 JWT 實現。

## 為什麼在身份驗證中使用依賴注入？

依賴注入為身份驗證提供了以下優勢：

- **可重用性**：在多個路由中重複使用相同的驗證邏輯
- **可測試性**：輕鬆模擬身份驗證過程進行測試
- **關注點分離**：將身份驗證邏輯與業務邏輯分離
- **靈活性**：可以根據不同需求組合多種身份驗證策略

## 基本身份驗證模式

### 1. API 密鑰驗證

最簡單的身份驗證形式，通過請求頭或查詢參數傳遞 API 密鑰。

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI()

# 創建 API 密鑰頭部驗證器
api_key_header = APIKeyHeader(name="X-API-Key")

# 定義依賴函數
async def get_api_key(api_key: str = Security(api_key_header)):
    # 在實際應用中，這裡應該查詢資料庫或配置
    valid_api_keys = {"your-api-key-here", "another-valid-key"}
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="無效的 API 密鑰"
        )
    
    return api_key

@app.get("/secure-endpoint")
async def secure_endpoint(api_key: str = Depends(get_api_key)):
    return {"message": "您已通過身份驗證", "key_used": api_key}
```

**優點**：
- 實現簡單
- 適合服務間通信

**缺點**：
- 安全性較低
- 無法輕鬆識別使用者身份

### 2. 基本身份驗證（Basic Auth）

使用用戶名和密碼進行身份驗證，通常通過 HTTP 基本身份驗證頭部傳遞。

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()

security = HTTPBasic()

# 定義依賴函數
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    # 在實際應用中，應該使用安全的密碼哈希比較
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "password123")
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="用戶名或密碼錯誤",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

@app.get("/users/me")
def read_current_user(username: str = Depends(get_current_user)):
    return {"username": username}
```

**優點**：
- 內建於 HTTP 協議
- 簡單易用

**缺點**：
- 每次請求都需要發送憑證
- 密碼以較弱的編碼方式傳輸

## 進階身份驗證模式

### 1. JWT 身份驗證

JSON Web Token (JWT) 是一種流行的身份驗證機制，特別適合無狀態 API。

```python
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# 配置
SECRET_KEY = "your-secret-key"  # 在實際應用中應使用安全的隨機密鑰
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 模型
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None

# 工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

# 模擬用戶資料庫
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "email": "johndoe@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
    }
}

# 輔助函數
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, fake_db[username]["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 依賴函數
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="無法驗證憑證",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="用戶已停用")
    return current_user

# 路由
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用戶名或密碼錯誤",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
```

**優點**：
- 無狀態，不需要在伺服器存儲會話
- 可包含用戶相關信息
- 可設置過期時間

**缺點**：
- 需要妥善保管密鑰
- Token 一旦發出無法撤銷（除非使用黑名單）

### 2. OAuth2 身份驗證

OAuth2 是一種授權框架，允許第三方應用訪問用戶資源而無需共享密碼。

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel

app = FastAPI()

# 配置 OAuth2 授權碼流
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://example.com/oauth/authorize",
    tokenUrl="https://example.com/oauth/token",
)

class User(BaseModel):
    username: str
    email: str
    full_name: str

# 模擬 OAuth2 令牌驗證
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 在實際應用中，這裡應該驗證令牌並從身份提供者獲取用戶信息
    if token != "valid_token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的身份驗證憑證",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 模擬從令牌中獲取的用戶信息
    return User(
        username="johndoe",
        email="johndoe@example.com",
        full_name="John Doe"
    )

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

**優點**：
- 支持第三方身份驗證
- 用戶無需向你的應用提供密碼
- 可以限制訪問範圍

**缺點**：
- 實現較為複雜
- 需要外部身份提供者

## 組合與自定義身份驗證

### 1. 多層身份驗證

結合多種身份驗證方式，提供更靈活的安全策略。

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from typing import Optional

app = FastAPI()

# 定義多種身份驗證方式
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# 組合身份驗證依賴
async def get_current_user(
    api_key: Optional[str] = Security(api_key_header),
    token: Optional[str] = Security(oauth2_scheme)
):
    if api_key:
        # 驗證 API 密鑰
        if api_key == "valid-api-key":
            return {"user": "api_user", "auth_method": "api_key"}
    
    if token:
        # 驗證 OAuth2 令牌
        if token == "valid-token":
            return {"user": "oauth_user", "auth_method": "oauth"}
    
    raise HTTPException(
        status_code=401,
        detail="需要有效的身份驗證憑證"
    )

@app.get("/secure")
async def secure_endpoint(user = Depends(get_current_user)):
    return {"message": "已通過身份驗證", "user": user}
```

**優點**：
- 支持多種身份驗證方式
- 提供更靈活的訪問控制

**缺點**：
- 增加了複雜性
- 需要更多的測試覆蓋

### 2. 基於角色的訪問控制 (RBAC)

在身份驗證之上添加授權邏輯，根據用戶角色限制訪問。

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from typing import List

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模擬用戶資料庫
fake_users_db = {
    "alice": {
        "username": "alice",
        "roles": ["user", "admin"]
    },
    "bob": {
        "username": "bob",
        "roles": ["user"]
    }
}

# 身份驗證依賴
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 模擬令牌解析，實際應用中應驗證 JWT
    username = token
    
    if username not in fake_users_db:
        raise HTTPException(status_code=401, detail="無效的憑證")
    
    return fake_users_db[username]

# 角色檢查依賴
def has_role(required_roles: List[str]):
    async def role_checker(current_user = Depends(get_current_user)):
        for role in required_roles:
            if role not in current_user["roles"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"權限不足，需要角色: {required_roles}"
                )
        return current_user
    
    return role_checker

# 路由
@app.get("/users/profile")
async def read_profile(user = Depends(get_current_user)):
    return user

@app.get("/admin/dashboard")
async def admin_dashboard(user = Depends(has_role(["admin"]))):
    return {"message": "管理員儀表板", "user": user}
```

**優點**：
- 精細的訪問控制
- 基於用戶角色的授權

**缺點**：
- 需要維護角色與權限映射
- 可能需要更複雜的角色層次結構

## 最佳實踐

1. **使用安全的密碼存儲**：始終使用如 bcrypt 或 Argon2 等安全的哈希算法存儲密碼。

2. **實施令牌過期**：為 JWT 或其他令牌設置合理的過期時間。

3. **使用 HTTPS**：所有身份驗證通信應通過 HTTPS 進行，以防止中間人攻擊。

4. **實施速率限制**：限制身份驗證嘗試次數，防止暴力破解攻擊。

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/token")
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # 身份驗證邏輯
    pass
```

5. **分層身份驗證**：將身份驗證邏輯分為多個可重用的依賴。

```python
# 基本身份驗證
async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 驗證令牌並返回用戶
    pass

# 確保用戶處於活動狀態
async def get_active_user(current_user = Depends(get_current_user)):
    if current_user["disabled"]:
        raise HTTPException(status_code=400, detail="用戶已停用")
    return current_user

# 確保用戶具有特定角色
async def get_admin_user(current_user = Depends(get_active_user)):
    if "admin" not in current_user["roles"]:
        raise HTTPException(status_code=403, detail="需要管理員權限")
    return current_user
```

6. **使用 OAuth2 密碼流時的安全考慮**：如果使用密碼流，確保客戶端是可信的。

## 結論

FastAPI 的依賴注入系統為實現各種身份驗證策略提供了強大而靈活的基礎。從簡單的 API 密鑰到複雜的 OAuth2 實現，依賴注入使得身份驗證邏輯可以被清晰地組織、重用和測試。

通過選擇適合您應用需求的身份驗證策略，並遵循安全最佳實踐，您可以構建既安全又易於使用的 API。記住，身份驗證是安全的第一道防線，但它應該是整體安全策略的一部分，包括授權、加密和其他安全措施。
