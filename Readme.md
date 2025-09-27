# Flow Assistant Prototype

Flow Assistant は、要件定義書 (`docs/requiement.md`) をもとにしたオンデバイス実行想定のプロアクティブ支援エージェントの Python プロトタイプです。Collector → Preprocessor → Index → Trigger Engine → RAG Engine → Card UI → Action Executor → Learning/Reporting というパイプラインを単一プロセスで再現しています。

## セットアップ
- 追加ライブラリ不要（Python 標準ライブラリのみで動作）
- 必要に応じて仮想環境を用意してください

## 使い方
```
python main.py demo
```
- 要件定義書をインデックス化し、規定のサンプル文脈を順番に処理します
- 提示される Assist Card に対し `adopt` / `ignore` / `snooze` を入力すると学習とレポートが更新されます
- 実行後に日次・週次レポートのテキスト要約を表示します

## 主なモジュール
- `flow_assistant/models.py`: Document / ContextSnapshot / Suggestion / Episode のデータモデル
- `flow_assistant/preprocessor.py`: 簡易サマリー & 擬似ベクトル埋め込み生成
- `flow_assistant/index.py`: SQLite + JSON ベクトル格納による検索
- `flow_assistant/triggers.py`: 要件定義書「初期ルール例」を実装したトリガー群
- `flow_assistant/rag.py`: RAG 風のドキュメント取り込み & 検索
- `flow_assistant/pipeline.py`: 全体パイプラインとレポート出力

## 次のステップ例
1. 実環境のテレメトリ（アプリ/OS連携）を Collector に接続
2. NPU を用いた要約器・埋め込み器の差し替え
3. UI/アクション層をデスクトップアプリやブラウザ拡張へ展開
