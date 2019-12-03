import draw as drw

jetson_UART = "/dev/ttyTHS1"
drawer = drw.Drawer(jetson_UART)

try:
    test = [(69,99), (152,33)]
    print(drawer.read(1))
    drawer.draw(test)
    

except Exception as e:
    print(str(e))
    drawer.cleanup()
