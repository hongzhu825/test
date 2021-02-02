output_worker=model_output(model_path='./service_model/v2.pth.tar')
import grpc
path='./service_model/v2.pth.tar'
print('model_path',path)
T = TatamiLayter(path)
json_path = './tatami_solution.json'
info = "10.51.0.2:50065"

def run(room_json=None):
    with grpc.insecure_channel(
            '10.51.0.2:50065') as channel:  # 192.168.1.51 #localhost#192.168.1.51:50058#'10.51.0.2:50058'
        greeter_stub = layout_pb2_grpc.LayoutStub(channel)
        data=room_json
        data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        #print('data',data)
        request = layout_pb2.Request(data=data ,top_k=1,room_type=2,model_type=None)

        print("发送信息:", request)
        try:
            response = str(T.predict(request,context=None))
        except Exception as e:
            response = "Error:{}".format(e)
        print("接收信息:", response)

        return response
i=0
with open(json_path, 'r') as f:
    while True:
        ss = f.readline()
        if not ss:
            break
        i+=1
        if i>600:
            break
        #print(i)
        empty_room_data = json.loads(ss)
        room_json = empty_room_data
        room_json['dnaId']=67209
        run(room_json)

        break