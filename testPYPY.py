from datetime import datetime

i = 0

startTime = datetime.now()
while True :
    i += 1
    print(i)
    if i >= 100000 :
        endTime = datetime.now()
        break

print("1초당 프린트 {0}번".format((endTime-startTime)/100000))
