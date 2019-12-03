import draw as drw

jetson_UART = "/dev/ttyTHS1"
drawer = drw.Drawer(jetson_UART)
data = None

try:
    test = [(69,99), (152,33)]
    print(drawer.read(1))
    drawer.send_packet(test)
    

except Exception as e:
    print(str(e))
    drawer.cleanup()
