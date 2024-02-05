from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import data_processing as dp

app = FastAPI()

class Item(BaseModel):
    filePath: str

@app.post("/upload-csv/")
async def upload_csv(item: Item):
    # CSVファイルのパスを受け取る
    file_path = item.filePath
    # ここでファイルパスを用いて何らかの処理を行う
    print(f"Received file path: {file_path}")
    dp.generate_onnx_model(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)