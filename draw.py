import time
import serial
import struct

class Drawer:
        def __init__(self, port=None, baudrate=115200):
                self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
        )

        def cleanup(self):
                self.serial_port.close()

        def read(self, size):
                return self.serial_port.read(size) 
        
        def send(self, data):
                data = struct.pack('I',data)
                self.serial_port.write(data)
                self.serial_port.flush()
                
        def draw(self, points):  
                '''points: [(x1, y1), (x2, y2)]'''
                for i, val in enumerate(points):
                    self.send(val[0])
                    self.send(val[1])
                    print('Sent', val)
                if self.read(1): # == b'\xff': #wait until we recieve something back
                    print('ACK')
                
