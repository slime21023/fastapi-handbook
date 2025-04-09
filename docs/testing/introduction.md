# 測試基礎

## FastAPI 測試簡介

FastAPI 是一個現代、快速、高效能的 Python Web 框架，專為 API 開發而設計，其內建的功能使得測試變得相對直接和高效。FastAPI 基於 Starlette 和 Pydantic，這兩個組件都有良好的測試支持，使得 FastAPI 應用程序的測試更加便捷。

在 FastAPI 中進行測試，我們主要關注以下幾個方面：
- API 端點的功能測試
- 請求和響應模型的驗證
- 依賴注入系統的行為
- 錯誤處理和異常情況
- 中間件和背景任務

FastAPI 提供了 `TestClient` 類，這是基於 `httpx` 庫的客戶端，可以模擬對 API 的請求，並檢查響應，而無需實際啟動服務器。

```python
from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
```

## 測試的重要性與優勢

### 為什麼測試很重要？

| 優勢 | 說明 |
|------|------|
| **品質保證** | 測試確保你的代碼按預期工作，並在發布前捕獲錯誤 |
| **重構信心** | 有了良好的測試覆蓋，你可以更自信地修改和改進代碼 |
| **文檔作用** | 測試可以作為代碼功能的活文檔，展示如何使用 API |
| **協作效率** | 測試幫助團隊成員理解代碼的預期行為，促進協作 |
| **持續集成** | 自動化測試是 CI/CD 流程的關鍵組成部分 |

### FastAPI 測試的特殊優勢

| 優勢 | 說明 |
|------|------|
| **類型安全** | FastAPI 的類型提示和 Pydantic 模型使得測試更加精確 |
| **自動文檔** | 測試可以驗證 API 文檔的準確性 |
| **依賴注入** | FastAPI 的依賴注入系統使得模擬和測試隔離變得簡單 |
| **非同步支持** | 可以直接測試非同步代碼，無需特殊處理 |
| **OpenAPI 驗證** | 可以測試 API 是否符合 OpenAPI 規範 |

## 測試策略與測試金字塔

### 測試金字塔

測試金字塔是一種視覺化模型，描述了不同類型測試的比例關係：

| 測試類型 | 層級 | 數量 | 特點 | FastAPI 應用 |
|---------|------|------|------|-------------|
| **單元測試** | 底層 | 最多 | 測試最小的代碼單元<br>執行快速，隔離性好 | 測試路由函數<br>測試依賴項<br>測試工具函數 |
| **整合測試** | 中層 | 適中 | 測試多個組件協同工作<br>可能包括外部依賴 | 使用 TestClient 測試 API 端點<br>測試數據庫交互<br>測試外部服務模擬 |
| **端到端測試** | 頂層 | 最少 | 測試整個應用程序流程<br>在真實環境中運行 | 測試完整用戶流程<br>測試前後端交互<br>測試系統整體功能 |

### 有效的測試策略

| 策略 | 說明 | 適用場景 |
|------|------|---------|
| **測試驅動開發 (TDD)** | 先寫測試，再實現功能 | 特別適合 FastAPI 的聲明式風格 |
| **行為驅動開發 (BDD)** | 基於用戶行為和需求編寫測試 | 使用 pytest-bdd 等工具實現 |
| **混合方法** | 單元測試採用 TDD<br>整合測試採用 BDD | 根據項目需求靈活調整 |

## FastAPI 測試工具概述

### 核心測試工具

| 工具 | 核心概念 | 主要用途 |
|------|---------|---------|
| **pytest** | 強大的 Python 測試框架 | 提供測試發現、執行和報告<br>支持 fixture、參數化測試和標記<br>擁有豐富的插件生態系統 |
| **FastAPI TestClient** | 基於 httpx 的測試客戶端 | 無需啟動服務器即可測試 API 端點<br>模擬 HTTP 請求和檢查響應<br>支持同步和非同步測試方式 |
| **pytest-asyncio** | 非同步測試擴展 | 支持測試非同步函數和協程<br>提供非同步 fixture 功能<br>與 FastAPI 的非同步本質完美匹配 |

### 輔助測試工具

| 工具 | 核心概念 | 主要用途 |
|------|---------|---------|
| **pytest-cov** | 代碼覆蓋率分析 | 測量測試覆蓋的代碼比例<br>生成覆蓋率報告<br>幫助識別未測試的代碼區域 |
| **mock / unittest.mock** | 對象模擬 | 創建和管理模擬對象<br>隔離被測代碼的外部依賴<br>控制和驗證函數調用 |
| **pytest-xdist** | 並行測試執行 | 加速測試套件運行<br>分配測試到多個 CPU 核心<br>支持分布式測試執行 |
| **factory_boy / faker** | 測試數據生成 | 創建測試模型實例<br>生成隨機但合理的測試數據<br>減少測試代碼中的重複數據定義 |

## 設置測試環境

### 項目結構

一個良好組織的 FastAPI 項目測試結構可能如下：

```
my_fastapi_app/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   ├── models/
│   └── services/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # 共享的 pytest fixtures
│   ├── test_main.py          # 主應用測試
│   ├── api/                  # API 端點測試
│   ├── models/               # 模型測試
│   └── services/             # 服務層測試
└── pytest.ini                # pytest 配置
```

### pytest 配置

典型的 `pytest.ini` 文件：

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
asyncio_mode = auto
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

### 測試依賴管理

建議在 `conftest.py` 中定義共享的測試固件，用於：

- 創建測試數據庫連接
- 設置測試客戶端
- 覆蓋依賴注入
- 提供測試數據
- 管理測試前後的清理工作

## 測試執行與報告

### 常用測試命令

| 命令類型 | 用途 | 示例命令 |
|---------|------|---------|
| **基本執行** | 運行測試 | `pytest`<br>`pytest tests/test_users.py`<br>`pytest tests/test_users.py::test_create_user` |
| **選擇性執行** | 基於標記或表達式選擇測試 | `pytest -m integration`<br>`pytest -k "user and not delete"` |
| **效率優化** | 提高測試執行效率 | `pytest -n auto`<br>`pytest --lf` (只運行上次失敗的測試) |
| **覆蓋率分析** | 檢查代碼覆蓋情況 | `pytest --cov=app --cov-report=html` |

### 測試報告選項

| 報告類型 | 用途 | 命令選項 |
|---------|------|---------|
| **控制台輸出** | 調整測試結果顯示方式 | `-v` (詳細)<br>`--durations=10` (顯示最慢的測試) |
| **結構化報告** | 生成機器可讀的報告格式 | `--junitxml=report.xml` (CI 系統常用) |
| **可視化報告** | 生成人類可讀的報告 | `--html=report.html` (需要 pytest-html 插件) |

### 常見的測試覆蓋率目標

| 覆蓋率類型 | 理想目標 | 最低要求 |
|-----------|---------|---------|
| 行覆蓋率 | > 90% | > 75% |
| 分支覆蓋率 | > 85% | > 70% |
| 函數覆蓋率 | > 95% | > 80% |

## 總結

FastAPI 測試基礎為你的 API 開發提供了堅實的框架。通過理解這些基本概念和工具，你可以建立一個全面的測試策略，確保你的 FastAPI 應用程序穩定、可靠且易於維護。

### 測試基礎要點

| 方面 | 關鍵點 |
|------|--------|
| **工具選擇** | 使用 pytest 作為主要測試框架<br>利用 FastAPI TestClient 測試 API 端點<br>選擇適合的輔助工具提高測試效率 |
| **測試策略** | 遵循測試金字塔原則<br>優先覆蓋核心業務邏輯<br>根據項目需求選擇適合的測試方法 |
| **環境設置** | 使用隔離的測試環境<br>合理組織測試文件結構<br>利用 fixtures 減少重複代碼 |
| **持續改進** | 定期檢查測試覆蓋率<br>重構測試以提高可維護性<br>將測試集成到 CI/CD 流程中 |

在接下來的章節中，我們將深入探討各種測試類型的具體實現，從單元測試到整合測試，以及如何處理數據庫、非同步代碼和模擬外部依賴。

---

通過這些基礎知識，你已經準備好開始為你的 FastAPI 應用程序編寫高質量的測試了。記住，好的測試不僅僅是捕獲錯誤，它們還是你的 API 設計和功能的指南。