import threading
import configparser
import signal
import sys
import time
import ray

ray.init()

config = configparser.ConfigParser()
config.read("config.ini")

DATASET_PATH  = str  (config["DEFAULT"]["DATASET_PATH" ])


#def signal_handler(signal, frame):
#    print("script killed...")
#    sys.exit(0)
#signal.signal(signal.SIGINT, signal_handler)


ds = Dataset(folder_path=DATASET_PATH, mem_length=10000)
ds.update_dataset(batch_size=256)

def dataset_updater():
    while True:
        #ds.update_dataset(batch_size=256)
        print("thread is working")
        time.sleep(1)

thread = threading.Thread(target=dataset_updater)
thread.daemon = True
thread.start()


while True:
	time.sleep(0.5)
	print("main program loop...")
