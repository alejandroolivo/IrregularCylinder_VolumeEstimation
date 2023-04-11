import sys
import socket
import threading
import numpy as np

class BcnWrapper(object):
    def __init__(self, IsParallelMode = True):
        """
        Constructor
            :param IsParallelMode: if True, run_method will be run in a new thread
        """

        try:
            # Checking args
            if len(sys.argv) == 1:
                print("No args")
                sys.exit(0)

            # instances
            self.ip = "127.0.0.1"
            self.receive_port = int(sys.argv[1])
            self.send_port = int(sys.argv[2])
            self.IsParallelMode = IsParallelMode

            # create udp socket
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp.bind((self.ip, self.receive_port))

            # send ready message
            self.udp.sendto(b"Ready", (self.ip, self.send_port))

            # signal to stop the thread
            self.IsRunning = True
            
            print("Ready")

            # main thread loop
            while(self.IsRunning):

                # wait for a message
                message = self.wait_for_message().decode("utf-8")

                # print
                print("Received: " + message)

                if self.IsParallelMode:
                    # run_method in a new thread
                    threading.Thread(target=self.run_method, args=(message,)).start()
                else:
                    self.run_method(message)

        except Exception as err:
            print(err)
            pass

    def run_method(self, message):
        """
        run a method
            :param message: message to run
        """

        # split message
        split_message = message.split('|')

        # call method if exist
        if split_message[0] in dir(self):
            getattr(self, split_message[0])(split_message[1])



    def wait_for_message(self):
        """
        wait for a new udp message
            :return: received message
        """
        #  Wait for next request from client
        message, _ = self.udp.recvfrom(1024)

        print(message)

        return message

    def send_message(self, tag, value):
        """
        send a message
            :param message: message to send
        """

        # send a message
        self.udp.sendto(str(tag +'|' + value).encode("utf-8"), (self.ip, self.send_port))

    def write_result(self, value:str, result:np):
        """
        write a result
            :param value: value to write
            :param result: result to write
        
        Example:
            self.write_result(value, np.arange(0, 4, 1, dtype=np.uint8))
            """

        # serialize numpy array to byte array
        with open(value, 'wb') as f:
            f.write(result.tobytes())