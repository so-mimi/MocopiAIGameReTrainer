from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import CreateModel as cm
from fastapi.responses import FileResponse
import config
import numpy as np
import os
import socket
import json
import threading

app = FastAPI()

class Item(BaseModel):
    filePath: str

@app.post("/upload-csv/")
async def upload_csv(item: Item):
    # CSVファイルのパスを受け取る
    file_path = item.filePath
    config.parent_dir = os.path.dirname(file_path)
    # ここでファイルパスを用いて何らかの処理を行う
    print(f"Received file path: {file_path}")
    config.MocopiGameAIModel = await cm.train_and_evaluate_model_async(file_path)
    return {"message": "File received successfully"}

@app.post("/reset/")
async def reset_system():
    # ここでリセット処理を実行
    print("Resetting system...")
    # configやその他のグローバル変数のリセット処理をここに記述
    reset_server_state()  # サーバーの状態をリセットする関数
    return {"message": "System reset successfully"}

class PredictItem(BaseModel):
    data: list[float]  # 推論に使用するデータのリスト


class ResetCommand(BaseModel):
    # 必要に応じてリセット処理に関するパラメータをここに定義
    pass

# UDPサーバー設定
UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 6789

def start_udp_server():
    global running
    running = True
    serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
    print(f"UDP server up and listening at {UDP_IP_ADDRESS}:{UDP_PORT_NO}")

    while running:
        data, addr = serverSock.recvfrom(65535)

        try:
            # JSON形式の文字列をPythonオブジェクトに変換
            item = json.loads(data.decode('utf-8'))
            if config.MocopiGameAIModel is None:
                print("Model is not trained yet.")
                continue
            
            input_data = np.array(item["data"]).reshape(1, -1)
            prediction = config.MocopiGameAIModel.predict(input_data)
            response = {"prediction": prediction.tolist()}
            
            # 応答をJSON形式の文字列に変換してクライアントに送信
            response_data = json.dumps(response).encode('utf-8')
            response_addr = (addr[0], 6790)  # addr[0]はクライアントのIPアドレス、6790はUnity側の受信ポート
            serverSock.sendto(response_data, response_addr)
            print(f"Prediction sent: {response} to {response_addr}")
            print(f"Prediction sent: {response} to {addr}")
        except Exception as e:
            print("An error occurred:", e)
            if not running:
                break 


def reset_server_state():
    global running
    # サーバーの状態をリセットする処理を実装
    print("Server state is being reset...")
    # モデルを初期化など...
    config.MocopiGameAIModel = None
    print("Server state reset successfully.")
    config.parent_dir = None
    # UDPサーバーのループを終了させる
    running = False


def restart_udp_server():
    global running
    running = True  # サーバーの実行状態を再びTrueに設定
    # UDPサーバースレッドを再起動
    thread = threading.Thread(target=start_udp_server)
    thread.start()
    print("UDP server restarted.")


# UDPサーバーを別スレッドで起動する関数
def run_udp_server():
    threading.Thread(target=start_udp_server).start()


if __name__ == "__main__":
    run_udp_server()
    uvicorn.run(app, host="127.0.0.1", port=8000)