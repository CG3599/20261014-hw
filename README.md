# Feature Lab — 50 Startups

五種特徵選擇方法（EFS、RFECV、LassoCV、Backward、SFS）的互動式結果網站。

網站原始碼位於 `docs/`，可直接用 GitHub Pages 從 `main` 分支的 `/docs` 發布，無需建置或外部 JavaScript 套件。

## 網頁功能

- 切換五種方法，查看 1–5 個特徵的組合與測試 RMSE / R²。
- 點選圖表圓點、結果列或特徵數滑桿。
- 疊加五種方法的曲線，下載 25 筆結果 CSV。
- URL 保留方法、特徵數與比較模式，可分享特定視圖。
- 顯示自動選擇方法的原始 nested CV 比較。

## 重現計算

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python main.py
```

請使用虛擬環境中的 Python。資料來源為 `50_Startups.csv`。資料共 50 筆，80/20 訓練測試切分，random_state=42。

固定數量的 RFECV 比較實際使用 RFE 路徑，保留原始未標準化係數設定；LassoCV 依標準化係數排名，同分時以標準化 OLS 係數排序，再重訓 OLS；Backward 強制移除至指定數量。這些曲線為診斷結果，不是原本自動選取的最佳特徵數。網頁不重新訓練模型。

`docs/data.js` 保存展示用的結果；Python 計算輸出在 `results/`。
