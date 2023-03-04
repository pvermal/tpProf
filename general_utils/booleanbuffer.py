import numpy as np
import os


class BooleanBuffer(object):
    def __init__(capacity):
        self.capacitiy = capacity  # indicates the maximum size of the buffer
        self.tail = -1  # points to the last item
        self.head = 0  # points to the first item
        self.actualSize = 0  # indicates the actual size of the buffer
        self.data = np.array(
            capacity, dtype=bool
        )  # array with the values of the buffer

    def Enqueue(self, item):
        if self.actualSize == self.capacity:
            raise Exception("Buffer full. First Dequeue")

        else:
            self.tail = (self.tail + 1) % self.capacity
            self.data[self.tail] = item
            self.actualSize = self.actualSize + 1

    def Dequeue(self):
        if self.actualSize == 0:
            print("Buffer empty. First Enqueue some item")
            return None

        else:
            tmp = self.data[self.head]
            self.head = (self.head + 1) % self.capacity
            self.actualSize = self.actualSize - 1

        return tmp

    def DequeueWithOffsetFromHead(self, offset):
        if self.actualSize == 0:
            print("Buffer empty. First Enqueue some item")
            return None
        # verify the case when (self.head + offset) > self.actualsize

        else:
            index = (self.head + offset) % self.capacity
            tmp = self.data[index]
            np.delete(self.data, index)
            np.append(self.data, True)
            self.trail -= 1
            self.actualSize = self.actualSize - 1

        return tmp

    def SwitchValuesFromTailToLastDifferent(self, offset):
        if self.actualSize == 0:
            return

        valueToSwitch = self.data[self.trail]
        self.data[self.trail] = ~(self.data[self.trail])
        index = (self.trail - 1) % self.capacity
        # verify the case when (self.head + offset) > self.actualsize
        if (
            index != self.trail
            and index < self.actualSize
            and self.data[index] == valueToSwitch
        ):
            self.data[index] = ~(self.data[index])
            index = (self.trail - 1) % self.capacity

        return

    def getValueWithOffsetFromTail(self):
        if self.actualSize == 0:
            print("Buffer empty. First Enqueue some item")
            return None

        else:
            tmp = self.data[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size = self.size - 1

        return tmp

    def getOutputSignal(self):
        return self.outputSignal

    def getVehicleCount(self):
        return len(self.vehicleList)

    def updateOutputSignal(self, frameTime):
        self.outputSignal.append((frameTime, int(self.isOccupiedNow)))

    def updateVehicleList(self, vehicleId):
        self.vehicleList.add(vehicleId)

    def updateIsOccupied(self, isOccupied, id):
        # First, update the "Prev" variables
        self.isOccupiedPrev = self.isOccupiedNow
        self.occupyIdPrev = self.occupyIdNow
        # Then, update the "Now" variables
        self.isOccupiedNow = isOccupied
        self.occupyIdNow = id
