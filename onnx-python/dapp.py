from os import environ
import numpy as np
import onnxruntime as ort
import logging
import requests



logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

rollup_server = environ["ROLLUP_HTTP_SERVER_URL"]
logger.info(f"HTTP rollup_server url is {rollup_server}")


# Load the ONNX model
model_path = "simple_nn.onnx"
session = ort.InferenceSession(model_path)
# Get the input and output names
input_names = [session.get_inputs()[0].name]
output_names = [session.get_outputs()[0].name]
# Run inference
outputs = session.run(output_names, {input_names[0]: np.random.randn(1, 10).astype('f')})

def handle_advance(data):
    logger.info(f"Received advance request data {data}")
    status = "accept"
    try:
        response = requests.post(
                rollup_server + "/notice", json={"payload": str({"modelOutputs": str(outputs[0][0][0])})}
        )
        logger.info(
            f"Received notice status {response.status_code} body {response.content}"
        )
    except Exception as e:
        logger.error(f"Exception while handling advance:{e}")

    return status


def handle_inspect(data):
    logger.info(f"Received inspect request data {data}")
    return "accept"


handlers = {
    "advance_state": handle_advance,
    "inspect_state": handle_inspect,
}

finish = {"status": "accept"}

while True:
    logger.info("Sending finish")
    response = requests.post(rollup_server + "/finish", json=finish)
    logger.info(f"Received finish status {response.status_code}")
    if response.status_code == 202:
        logger.info("No pending rollup request, trying again")
    else:
        rollup_request = response.json()
        data = rollup_request["data"]
        handler = handlers[rollup_request["request_type"]]
        finish["status"] = handler(rollup_request["data"])
