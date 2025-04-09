# 單元測試

## 單元測試的基本概念

單元測試是測試金字塔的基礎層，專注於測試代碼的最小單元（通常是函數或方法）。在 FastAPI 應用程序中，單元測試主要針對不依賴於 HTTP 請求/響應流程的獨立組件。

| 概念 | 說明 |
|------|------|
| **測試範圍** | 單一函數、方法或類的行為 |
| **隔離性** | 與其他組件和外部系統完全隔離 |
| **執行速度** | 非常快速，通常毫秒級 |
| **依賴處理** | 使用模擬(mock)或存根(stub)替代外部依賴 |
| **數量比例** | 在測試金字塔中佔比最大，約 70-80% |

## FastAPI 中的單元測試目標

在 FastAPI 應用中，以下組件是單元測試的主要目標：

| 組件類型 | 測試重點 | 示例 |
|---------|---------|------|
| **工具函數** | 輸入/輸出轉換、格式化、計算 | 日期轉換、金額計算、字符串處理 |
| **業務邏輯** | 核心業務規則和算法 | 折扣計算、資格驗證、狀態轉換 |
| **Pydantic 模型** | 數據驗證和轉換 | 模型實例化、驗證、轉換方法 |
| **依賴函數** | 提供給路由的可注入依賴 | 權限檢查、參數驗證 |
| **自定義異常** | 異常行為和屬性 | 錯誤碼、錯誤消息格式 |

## 單元測試的最佳實踐

### 命名和組織

| 實踐 | 說明 |
|------|------|
| **一致的命名** | 使用描述性名稱，如 `test_calculate_discount_with_valid_input` |
| **按模塊組織** | 測試文件結構應反映應用結構，如 `test_utils.py` 對應 `utils.py` |
| **測試類分組** | 相關測試可以組織在測試類中，如 `TestUserService` |
| **功能分類** | 按功能或場景分類測試，如 `test_validation_cases`, `test_error_cases` |

### 測試設計原則

| 原則 | 說明 |
|------|------|
| **單一職責** | 每個測試只測試一個行為或功能點 |
| **獨立性** | 測試之間不應有依賴關係或執行順序要求 |
| **確定性** | 測試結果應該是可預測的，不受環境變化影響 |
| **邊界測試** | 測試邊界條件和極端情況 |
| **錯誤案例** | 不僅測試正常流程，也要測試錯誤處理 |
| **代碼覆蓋** | 確保測試覆蓋所有代碼路徑和分支 |

### AAA 模式

單元測試通常遵循 Arrange-Act-Assert (AAA) 模式：

| 階段 | 目的 | 內容 |
|------|------|------|
| **Arrange** | 設置測試環境 | 創建對象、設置參數、準備輸入數據 |
| **Act** | 執行被測試的行為 | 調用被測函數或方法 |
| **Assert** | 驗證結果 | 檢查返回值、狀態變化或異常 |

## 工具函數的單元測試

工具函數通常是最容易進行單元測試的組件，因為它們往往是純函數（給定相同輸入總是產生相同輸出，沒有副作用）。

### 測試策略

| 策略 | 說明 |
|------|------|
| **參數化測試** | 使用多組輸入/輸出數據測試同一函數 |
| **邊界值分析** | 測試函數在邊界條件下的行為 |
| **異常處理** | 驗證函數對無效輸入的處理 |
| **性能檢查** | 對於關鍵工具函數，可以測試性能表現 |

### 示例：日期處理函數測試

假設我們有一個將字符串轉換為日期的工具函數：

```python
# app/utils/date_utils.py
from datetime import datetime, date
from typing import Optional

def parse_date_string(date_str: str) -> Optional[date]:
    """將字符串解析為日期對象，支持多種格式"""
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%Y.%m.%d"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None
```

對應的單元測試：

```python
# tests/utils/test_date_utils.py
import pytest
from datetime import date
from app.utils.date_utils import parse_date_string

@pytest.mark.parametrize("date_str, expected", [
    ("2023-01-15", date(2023, 1, 15)),  # ISO 格式
    ("15/01/2023", date(2023, 1, 15)),  # 歐洲格式
    ("2023.01.15", date(2023, 1, 15)),  # 點分隔格式
    ("invalid-date", None),             # 無效格式
    ("", None),                         # 空字符串
])
def test_parse_date_string(date_str, expected):
    # Act
    result = parse_date_string(date_str)
    
    # Assert
    assert result == expected
```

## 業務邏輯的單元測試

業務邏輯是應用的核心，需要全面的測試覆蓋。

### 測試策略

| 策略 | 說明 |
|------|------|
| **場景測試** | 測試不同業務場景下的邏輯行為 |
| **規則驗證** | 確保業務規則得到正確實施 |
| **狀態轉換** | 測試狀態機或工作流邏輯 |
| **依賴模擬** | 使用 mock 隔離外部依賴 |

### 示例：折扣計算服務測試

假設我們有一個計算訂單折扣的服務：

```python
# app/services/discount_service.py
from decimal import Decimal
from typing import Optional

class DiscountService:
    def calculate_discount(
        self, 
        order_total: Decimal, 
        user_tier: str, 
        coupon_code: Optional[str] = None
    ) -> Decimal:
        """計算訂單折扣金額"""
        # 基礎折扣率
        discount_rate = Decimal('0.0')
        
        # 會員等級折扣
        if user_tier == "gold":
            discount_rate += Decimal('0.05')
        elif user_tier == "platinum":
            discount_rate += Decimal('0.1')
            
        # 優惠券折扣
        if coupon_code == "SAVE10":
            discount_rate += Decimal('0.1')
        elif coupon_code == "SAVE20":
            discount_rate += Decimal('0.2')
        
        # 最大折扣率為 30%
        discount_rate = min(discount_rate, Decimal('0.3'))
        
        return order_total * discount_rate
```

對應的單元測試：

```python
# tests/services/test_discount_service.py
import pytest
from decimal import Decimal
from app.services.discount_service import DiscountService

class TestDiscountService:
    @pytest.fixture
    def service(self):
        return DiscountService()
    
    @pytest.mark.parametrize("order_total, user_tier, coupon_code, expected_discount", [
        # 基本測試案例
        (Decimal('100.00'), "regular", None, Decimal('0.00')),
        (Decimal('100.00'), "gold", None, Decimal('5.00')),
        (Decimal('100.00'), "platinum", None, Decimal('10.00')),
        
        # 優惠券測試
        (Decimal('100.00'), "regular", "SAVE10", Decimal('10.00')),
        (Decimal('100.00'), "regular", "SAVE20", Decimal('20.00')),
        (Decimal('100.00'), "regular", "INVALID", Decimal('0.00')),
        
        # 組合折扣測試
        (Decimal('100.00'), "gold", "SAVE10", Decimal('15.00')),
        (Decimal('100.00'), "platinum", "SAVE20", Decimal('30.00')),  # 最大折扣率 30%
        
        # 金額測試
        (Decimal('0.00'), "platinum", "SAVE20", Decimal('0.00')),
        (Decimal('1000.00'), "gold", "SAVE10", Decimal('150.00')),
    ])
    def test_calculate_discount(
        self, service, order_total, user_tier, coupon_code, expected_discount
    ):
        # Act
        discount = service.calculate_discount(order_total, user_tier, coupon_code)
        
        # Assert
        assert discount == expected_discount
```

## Pydantic 模型的單元測試

Pydantic 模型是 FastAPI 應用的重要組成部分，負責數據驗證和轉換。

### 測試策略

| 策略 | 說明 |
|------|------|
| **實例化驗證** | 測試模型能否正確實例化 |
| **字段驗證** | 測試字段約束和驗證邏輯 |
| **默認值** | 驗證默認值是否正確設置 |
| **轉換方法** | 測試自定義的轉換方法 |
| **錯誤處理** | 驗證無效數據的錯誤信息 |

### 示例：用戶模型測試

假設我們有一個用戶模型：

```python
# app/models/user.py
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import date
from typing import Optional

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    birth_date: Optional[date] = None
    password: str = Field(..., min_length=8)
    password_confirm: str
    
    @validator('password_confirm')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('密碼不匹配')
        return v
    
    def to_user_model(self):
        """轉換為用戶模型（不包含確認密碼）"""
        data = self.dict(exclude={'password_confirm'})
        return data
```

對應的單元測試：

```python
# tests/models/test_user.py
import pytest
from pydantic import ValidationError
from datetime import date
from app.models.user import UserCreate

class TestUserCreateModel:
    def test_valid_user_data(self):
        # Arrange
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "birth_date": "1990-01-01",
            "password": "securepass",
            "password_confirm": "securepass"
        }
        
        # Act
        user = UserCreate(**user_data)
        
        # Assert
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.birth_date == date(1990, 1, 1)
        assert user.password == "securepass"
    
    def test_password_validation(self):
        # Arrange
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "securepass",
            "password_confirm": "different"
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        
        errors = exc_info.value.errors()
        assert any(error["msg"] == "密碼不匹配" for error in errors)
    
    def test_username_length_validation(self):
        # Arrange - 用戶名太短
        user_data = {
            "username": "ab",  # 少於最小長度 3
            "email": "test@example.com",
            "password": "securepass",
            "password_confirm": "securepass"
        }
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            UserCreate(**user_data)
        
        errors = exc_info.value.errors()
        assert any("username" in error["loc"] for error in errors)
    
    def test_to_user_model(self):
        # Arrange
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "birth_date": "1990-01-01",
            "password": "securepass",
            "password_confirm": "securepass"
        }
        user = UserCreate(**user_data)
        
        # Act
        user_model_data = user.to_user_model()
        
        # Assert
        assert "password_confirm" not in user_model_data
        assert user_model_data["username"] == "testuser"
        assert user_model_data["email"] == "test@example.com"
        assert user_model_data["birth_date"] == date(1990, 1, 1)
        assert user_model_data["password"] == "securepass"
```

## 依賴函數的單元測試

依賴函數是 FastAPI 的重要特性，用於路由之間共享代碼和邏輯。

### 測試策略

| 策略 | 說明 |
|------|------|
| **獨立測試** | 將依賴函數視為普通函數進行測試 |
| **模擬請求上下文** | 模擬依賴函數在請求中的行為 |
| **異常處理** | 測試依賴函數的錯誤處理邏輯 |
| **返回值驗證** | 確保依賴函數返回正確的值 |

### 示例：簡單認證依賴測試

假設我們有一個簡單的認證依賴函數：

```python
# app/dependencies/auth.py
from fastapi import Header, HTTPException, status
from typing import Optional

def get_api_key(api_key: Optional[str] = Header(None)) -> str:
    """驗證 API 金鑰"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API 金鑰"
        )
    
    # 在實際應用中，這裡會檢查數據庫或配置
    valid_api_keys = ["test_key", "dev_key", "prod_key"]
    
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="無效的 API 金鑰"
        )
    
    return api_key
```

對應的單元測試：

```python
# tests/dependencies/test_auth.py
import pytest
from fastapi import HTTPException
from app.dependencies.auth import get_api_key

def test_get_api_key_valid():
    # Arrange
    valid_key = "test_key"
    
    # Act
    result = get_api_key(valid_key)
    
    # Assert
    assert result == valid_key

def test_get_api_key_missing():
    # Arrange & Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        get_api_key(None)
    
    assert exc_info.value.status_code == 401
    assert "缺少 API 金鑰" in exc_info.value.detail

def test_get_api_key_invalid():
    # Arrange
    invalid_key = "invalid_key"
    
    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        get_api_key(invalid_key)
    
    assert exc_info.value.status_code == 403
    assert "無效的 API 金鑰" in exc_info.value.detail
```

## 使用模擬 (Mock) 進行單元測試

在單元測試中，我們經常需要隔離外部依賴，如數據庫、外部 API 或文件系統。Python 的 `unittest.mock` 模塊提供了強大的模擬功能。

### 常用模擬技術

| 技術 | 用途 | 適用場景 |
|------|------|---------|
| **patch 裝飾器** | 替換模塊中的對象 | 模擬導入的函數或類 |
| **MagicMock** | 創建具有特定行為的模擬對象 | 模擬複雜對象和方法鏈 |
| **side_effect** | 定義模擬調用的行為 | 模擬異常或動態返回值 |
| **return_value** | 設置模擬調用的返回值 | 簡單的返回值模擬 |
| **assert_called_with** | 驗證模擬是否使用特定參數調用 | 驗證函數調用參數 |

### 示例：使用模擬的服務測試

假設我們有一個使用外部服務的產品服務：

```python
# app/services/product_service.py
from typing import Dict, List, Any
from app.repositories.product_repository import ProductRepository

class ProductService:
    def __init__(self, product_repo: ProductRepository):
        self.product_repo = product_repo
    
    def get_product_by_id(self, product_id: int) -> Dict[str, Any]:
        """根據 ID 獲取產品"""
        product = self.product_repo.get_by_id(product_id)
        if not product:
            raise ValueError(f"產品不存在: {product_id}")
        return product
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """獲取指定類別的所有產品"""
        return self.product_repo.get_by_category(category)
    
    def search_products(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索產品"""
        if not query or len(query) < 3:
            raise ValueError("搜索查詢必須至少包含 3 個字符")
        
        return self.product_repo.search(query, limit)
```

對應的單元測試：

```python
# tests/services/test_product_service.py
import pytest
from unittest.mock import Mock
from app.services.product_service import ProductService

@pytest.fixture
def mock_product_repo():
    return Mock()

@pytest.fixture
def product_service(mock_product_repo):
    return ProductService(mock_product_repo)

def test_get_product_by_id_success(product_service, mock_product_repo):
    # Arrange
    product_id = 1
    expected_product = {"id": product_id, "name": "測試產品", "price": 99.99}
    mock_product_repo.get_by_id.return_value = expected_product
    
    # Act
    result = product_service.get_product_by_id(product_id)
    
    # Assert
    assert result == expected_product
    mock_product_repo.get_by_id.assert_called_once_with(product_id)

def test_get_product_by_id_not_found(product_service, mock_product_repo):
    # Arrange
    product_id = 999
    mock_product_repo.get_by_id.return_value = None
    
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        product_service.get_product_by_id(product_id)
    
    assert f"產品不存在: {product_id}" in str(exc_info.value)
    mock_product_repo.get_by_id.assert_called_once_with(product_id)

def test_get_products_by_category(product_service, mock_product_repo):
    # Arrange
    category = "電子產品"
    expected_products = [
        {"id": 1, "name": "手機", "category": category},
        {"id": 2, "name": "平板電腦", "category": category}
    ]
    mock_product_repo.get_by_category.return_value = expected_products
    
    # Act
    result = product_service.get_products_by_category(category)
    
    # Assert
    assert result == expected_products
    mock_product_repo.get_by_category.assert_called_once_with(category)

def test_search_products_valid_query(product_service, mock_product_repo):
    # Arrange
    query = "手機"
    limit = 5
    expected_results = [{"id": 1, "name": "智能手機"}]
    mock_product_repo.search.return_value = expected_results
    
    # Act
    result = product_service.search_products(query, limit)
    
    # Assert
    assert result == expected_results
    mock_product_repo.search.assert_called_once_with(query, limit)

def test_search_products_invalid_query(product_service, mock_product_repo):
    # Arrange
    query = "ab"  # 少於 3 個字符
    
    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        product_service.search_products(query)
    
    assert "搜索查詢必須至少包含 3 個字符" in str(exc_info.value)
    mock_product_repo.search.assert_not_called()
```

## 單元測試的常見陷阱與解決方案

| 陷阱 | 問題 | 解決方案 |
|------|------|---------|
| **過度模擬** | 過多的模擬使測試變得脆弱且難以維護 | 只模擬外部依賴，不模擬被測系統的內部 |
| **測試實現而非行為** | 測試依賴於代碼的具體實現 | 專注於測試公共 API 和可觀察行為 |
| **測試覆蓋率迷思** | 過分追求高覆蓋率而忽略測試質量 | 關注關鍵路徑和邊界條件的測試 |
| **忽略邊界條件** | 只測試正常情況，忽略極端情況 | 系統地識別和測試邊界條件 |
| **測試不穩定** | 測試結果不一致或依賴環境 | 確保測試的確定性和獨立性 |
| **測試過於複雜** | 測試代碼比被測代碼更複雜 | 保持測試簡單，一個測試只測一個行為 |

## 總結

單元測試是 FastAPI 應用程序測試策略的基礎。通過有效地測試工具函數、業務邏輯、Pydantic 模型和依賴函數，你可以確保應用程序的核心組件按預期工作。

### 單元測試要點

| 方面 | 關鍵點 |
|------|--------|
| **測試範圍** | 專注於測試最小的獨立代碼單元<br>確保測試的隔離性和確定性 |
| **測試設計** | 遵循 AAA 模式組織測試<br>使用參數化測試減少重複 |
| **依賴處理** | 使用模擬隔離外部依賴<br>驗證與依賴的交互 |
| **測試覆蓋** | 確保測試覆蓋正常流程和錯誤情況<br>關注邊界條件和極端情況 |
| **代碼質量** | 保持測試代碼的簡潔和可讀性<br>避免測試代碼中的邏輯複雜性 |

在下一章節中，我們將探討如何使用 FastAPI 的 TestClient 進行 API 端點的整合測試，以及如何處理數據庫和其他外部依賴。

---

通過良好的單元測試實踐，你可以建立對代碼的信心，使重構和新功能開發變得更加安全和高效。記住，單元測試不僅是捕獲錯誤的工具，也是代碼設計的指南。