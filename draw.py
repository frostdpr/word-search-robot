import time
import serial

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

	def draw(self, p1 : (int, int), p2 : (int, int)):
		pass

	def read(self):
	    return self.serial_port.read(8) 
	
	def send(self, data):
	    self.serial_port.write(data)
