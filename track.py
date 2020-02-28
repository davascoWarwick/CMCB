import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy as np



class Track():
    # Members shared between all track objects

    def __init__(self, initialPoint, ellipseParama, ellipseParamb, trackID, trackStart):
        # Members private to each Track object

        # Define ID number for reference
        self.ID = trackID

        # Define positions and orientations:
        self.points = list()
        self.points.append(initialPoint)
        # DEBUG: print("Initialising track, first point is %s " % (str(initialPoint)))

        # Define shape:
        self.a = list()
        self.a.append(ellipseParama)
        self.b = list()
        self.b.append(ellipseParamb)

        # Define start time (frame number of entry) and array containing all times bound:
        self.start = trackStart
        self.time = list()
        self.time.append(trackStart)  # This variable is incremented as time passes

        return



    def addPoint(self, newpoint, ellipseparama, ellipseparamb, t):
        # Add point to existing track object

        self.points.append(newpoint)
        self.a.append(ellipseparama)
        self.b.append(ellipseparamb)
        self.time.append(t)

        return



    def trackLength(self):
        # Find length of track

        length = len(self.points)

        return length



    # Define functions to output members private to track object:
    def readXPoint(self):

        length = len(self.points)

        return self.points[length-1][0]



    def readYPoint(self):

        length = len(self.points)

        return self.points[length-1][1]



    def readTheta(self):

        length = len(self.points)

        return self.points[length-1][2] * (180. / math.pi)



    def reada(self):

        length = len(self.points)

        return self.a[length-1]



    def readAlla(self):

        return self.a



    def readb(self):

        length = len(self.points)

        return self.b[length-1]



    def readAllb(self):

        return self.b



    def readID(self):

        return self.ID



    def readStart(self):

        return self.start



    def readOngoingTime(self):

        length = len(self.time)

        return self.time[length-1]



    def readTime(self):

        return self.time



    def readAllXPoints(self):

        allX = list()
        for point in self.points:
            allX.append(point[0])

        return allX



    def readAllYPoints(self):

        allY = list()
        for point in self.points:
            allY.append(point[1])

        return allY



    def readAllThetaPoints(self):

        allTheta = list()
        for point in self.points:
            allTheta.append(point[2])

        return allTheta



    def dispSL(self):
        # Calculate final straight-line displacement from initial position in px (all tracks with a single point have a displacement of zero)

        length = len(self.points)
        displacement = math.sqrt(((self.points[length - 1][0] - self.points[0][0]) ** 2) + ((self.points[length - 1][1] - self.points[0][1]) ** 2))

        return displacement



    def dispCumu(self):
        # Calculate final cumulative displacement (path arc length) from initial position in px (all tracks with a single point have a displacement of zero)

        # Initialise variable to store cumulative displacement
        displacement = 0

        # Define variable to label loops
        i = 0

        # Loop over all points on track
        for point in self.points:

            # After first iteration only, once buffer array has been initialised
            if i > 0:

                # Sum displacements between timesteps
                displacement += math.sqrt(((point[0] - buff[0]) ** 2) + ((point[1] - buff[1]) ** 2))

            # Increment loop counter for labelling
            i += 1

            # Store current point to find displacements
            buff = point

        return displacement



    def processiveDefine(self, dt, dx, diffusivity):
        # Approximate whether track has processive region by finding large displacements (over 10 timesteps) that cannot be due to Brownian motion

        # Define track length
        length = len(self.points)

        # Define diffusivity in units of pixels and frames (for easier comparison)
        D = diffusivity

        # Define average displacement of diffusive object in 10 timesteps
        procLength = np.sqrt(4 * D * 10 * dt)

        # Define threshhold for processive motion as some multiple of this, define constant that varies degree of processivity required for detection (larger C -> fewer processive tracks)
        C = 2
        procLength *= C  # End up with procLength = 4 * np.sqrt(D * 10 * dt)

        # Define flag to test whether processive motion is observed in track
        proc = 0

        # Require a dwell time greater than 10 timesteps to define track as processive
        if length > 10:

            # Loop over all 10 timestep windows for tracked object
            for i in range(0, length - 10):

                # Define displacement during 10 timesteps
                disp = np.sqrt(((self.points[i + 10][0] - self.points[i][0]) ** 2) + ((self.points[i + 10][1] - self.points[i][1]) ** 2)) * dx

                # A large enough displacement cannot be due to diffusive motion
                if disp > procLength:

                    proc = 1

                    # Can break out of loop as track has at least one region of interest
                    break

        return proc



    def processiveFind(self, dt, dx, diffusivity, minLen):
        # Locate processive region(s) of track

        # Define track length
        length = len(self.points)

        # Define diffusivity in units of pixels and frames (for easier comparison)
        D = diffusivity

        # Locate processive regions by studying the correlation between displacement vectors between timesteps, require this to be greater than the noise in the correlation expected for a diffusive object
        qs = np.sqrt(8) * D * dt


        # Calculate displacement correlations to check whether regions of track are diffusive or processive:
        # Initialise arrays for analysis forwards in time (store displacements, correlations, average correlations, and binary array stating whether point is processive or not)
        dr = np.zeros((length, 2))
        pdt = np.zeros(length)
        qdt1 = np.zeros(length)
        b1 = np.zeros(length)

        # Define displacements for all possible timesteps
        for i in range(0, length - 1):

            dr[i][0] = (self.points[i + 1][0] - self.points[i][0]) * dx
            dr[i][1] = (self.points[i + 1][1] - self.points[i][1]) * dx

        # Define displacement correlations for all possible timesteps
        for i in range(0, length - 2):

            pdt[i] = (dr[i + 1][0] * dr[i][0]) + (dr[i + 1][1] * dr[i][1])

        # Define average of displacement correlations over 2 timesteps
        for i in range(0, length - 3):

            qdt1[i] = (pdt[i + 1] + pdt[i]) / 2

        # Define a region as processive if average of displacement correlations is greater than the noise in this value for a diffusing object
        for i in range(0, length - 3):

            if qdt1[i] > qs:

                # Store results as binary
                b1[i] = 1


        # Can repeat analysis backwards in time to account for time reversibility of the processive region
        b2 = np.zeros(length)

        backwards = 1
        if backwards == 1:

            dr = np.zeros((length, 2))
            pdt = np.zeros(length)
            qdt2 = np.zeros(length)

            for i in range(0, length - 1):

                dr[(length - 1) - i][0] = (self.points[(length - 1) - (i + 1)][0] - self.points[(length - 1) - i][0]) * dx
                dr[(length - 1) - i][1] = (self.points[(length - 1) - (i + 1)][1] - self.points[(length - 1) - i][1]) * dx

            for i in range(0, length - 2):

                pdt[(length - 1) - i] = (dr[(length - 1) - (i + 1)][0] * dr[(length - 1) - i][0]) + (dr[(length - 1) - (i + 1)][1] * dr[(length - 1) - i][1])

            for i in range(0, length - 3):

                qdt2[(length - 1) - i] = (pdt[(length - 1) - (i + 1)] + pdt[(length - 1) - i]) / 2

            # Convert results to binary
            for i in range(0, length - 3):

                if qdt2[(length - 1) - i] > qs:

                    b2[(length - 1) - i] = 1


        # Combine binary arrays
        b = np.zeros(length)

        # Loop over all points of track
        for i in range(0, length):

            # A point is processive if it was defined as processive when looking forwards or backwards in time
            if b1[i] == 1 or b2[i] == 1:

                b[i] = 1

        # DEBUG: Plot binary array that states whether object is moving processively at each point on the track
        # plt.plot(range(length), b)


        # Remove small processive regions from binary array (object must be moving processively for minLen timesteps for region to be defined as processive):
        # Loop over lengths of binary array equal to 1 (processive) that should be removed as they are less than minLen in length
        for i in range(2, minLen + 1):
            # DEBUG: Check indexes
            # print('New track, length = {}\ni = {}'.format(length, i))

            # Loop over all times that can be checked for processive motion for a given value of i
            for j in range(i, length):
                # DEBUG: Check indexes
                # print('j = {}'.format(j))

                # If current value (j) and value at time (j - i) of binary array are both zero, the array cannot have values of 1 (processive) between them, as this would give a processive region of length less than minLen
                if (b[j - i] == 0) and (b[j] == 0):

                    # Count number of processive points (1's) between these times
                    count = 0
                    for k in range(1, i):
                        # DEBUG: Check indexes
                        # print('k = {}'.format(k))

                        if b[j - k] == 1:
                            count += 1

                    # If all points between these times are processive, then set them equal to 0 (non-processive) instead
                    if count == i - 1:

                        for k in range(1, i):
                            # DEBUG: Check indexes
                            # print('k = {}'.format(k))

                            b[j - k] = 0


        # Repeat for boundary terms that need special treatment:
        for i in range(2, minLen + 1):

            # This time consider objects with processive regions starting at the boundaries (no initial zero value to count from)
            # If the number of processive points (1's) in binary array starting at boundary is less than minLen in length, they must be removed
            if b[0] == 1 and b[i - 1] == 0:

                for j in range(0, i - 1):

                    b[j] = 0

            # Repeat for other boundary
            if b[length - 1] == 1 and b[length - 1 - (i - 1)] == 0:

                for j in range(0, i - 1):

                    b[length - 1 - j] = 0


        # DEBUG: Check there are no processive regions of size less than minLen
        # print(b)
        j = 0
        while j < length:
            if b[j] == 1:
                count = 1
                for k in range(j + 1, length):
                    if b[k] == 1:
                        count += 1
                        if k == length - 1:
                            j = k + 1
                            if count < minLen:
                                print(count)
                            break
                    else:
                        j = k + 1
                        if count < minLen:
                            print(count)
                        break
            else:
                j += 1


        return (qdt1, qdt2, qs, b)



    def angles(self, fileDataOut, maxLen, b):
        # Find orientation angle of object and difference between orientation angle and direction of propagation at each time

        # Define length of track
        length = len(self.points)

        # Define lists to store angle of propagation (requires a displacement to define the angle, so only have (length - 1) data points)
        phi = np.zeros(length - 1)

        # Define lists to store orientation angle
        theta = np.zeros(length)
        ang = list()
        ang2 = list()

        # Define list to store difference between orientation angle and direction of propagation
        delAng = list()

        # Define buffer list for storage
        buff = list()

        # Define variable to label loops
        i = 0



        # Find propagation angle:
        # Loop over length of track in time
        for point in self.points:

            # After first iteration only, once buffer array has been initialised
            if i > 0:

                # Find unit vector for displacement between timesteps
                dx = point[0] - buff[0]
                dy = point[1] - buff[1]
                mag = math.sqrt((dy * dy) + (dx * dx))
                dx /= mag
                dy /= mag

                phi[i - 1] = math.atan2(dy, dx)

            # Increment loop counter for labelling
            i += 1

            # Store current point to find displacements
            buff = point

        # Reset loop counter
        i = 0



        # Find orientation angle:
        # First convert angle component of position array to full (2 * pi) domain by tracking motion from initial angle
        # Set initial angle
        theta[0] = self.points[0][2]

        # Find all subsequent orientations for theta assuming small changes in time
        for j in range(1, length):

            # If change in angle between previous theta value and current orientation angle is small, then angle at next time is equal to orientation provided by tracking
            if abs(self.points[j][2] - theta[j - 1]) <= math.pi / 2:

                theta[j] = self.points[j][2]

            # Large changes in angle of orientation provided by the tracking can be flipped by pi radians (due to elliptical symmetry) to allow for smaller angle change
            else:

                theta[j] = self.points[j][2]

                # Until change in angle is small (less than pi / 2) rotate orientation provided by tracking
                while abs(theta[j] - theta[j - 1]) > math.pi / 2:

                    if theta[j] > theta[j - 1]:

                        theta[j] = theta[j] - math.pi
                        # DEBUG: Check angle has been changed correctly
                        # print('Change at {}-: {:.4f}, {:.4f} -> {:.4f}'.format(j, theta[j - 1], self.points[j][2], theta[j]))

                    else:

                        theta[j] = theta[j] + math.pi
                        # DEBUG: Check angle has been changed correctly
                        # print('Change at {}+: {:.4f}, {:.4f} -> {:.4f}'.format(j, theta[j - 1], self.points[j][2], theta[j]))

        # Repeat for other possible initial orientation (due to elliptical symmetry and pi domain of orientation angle from tracking)
        theta2 = np.zeros(length)

        # Set initial angle as other possible orientation on (-pi < theta2 < pi) domain
        if theta[0] < 0:
            initial = theta[0] + math.pi
        else:
            initial = theta[0] - math.pi
        theta2[0] = initial

        for j in range(1, length):
            if abs(self.points[j][2] - theta2[j - 1]) <= math.pi / 2:
                theta2[j] = self.points[j][2]
            else:
                theta2[j] = self.points[j][2]
                while abs(theta2[j] - theta2[j - 1]) > math.pi / 2:
                    if theta2[j] > theta2[j - 1]:
                        theta2[j] = theta2[j] - math.pi
                    else:
                        theta2[j] = theta2[j] + math.pi



        # Choose between orientation angle trajectories by finding which initial orientation gives angle closest to propagation direction on average:
        # For track with no processive region set initial angle as closest to average propagation direction
        if sum(b) == 0:

            # Find average propagation direction in processive region(s) of track
            direction = np.zeros(length - 1)

            for j in range(0, length - 1):

                # First entry is initial propagation direction already calculated
                if j == 0:

                    direction[j] = phi[j]

                # Assume that propagation angle cannot change by more than pi in a single timestep (also due to equivalence of angles on circular domain)
                else:

                    direction[j] = phi[j]

                    # Until change in angle is small (less than pi) rotate propagation angle (use equivalent angle on circular domain)
                    if abs(direction[j] - direction[j - 1]) > math.pi:
                        while abs(direction[j] - direction[j - 1]) > math.pi:
                            if direction[j] > direction[j - 1]:
                                direction[j] = direction[j] - (2 * math.pi)
                            else:
                                direction[j] = direction[j] + (2 * math.pi)

            # Calculate average quantities to compare two orientation angle trajectories to propagation angle trajectory:
            thetaAv = 0
            theta2Av = 0

            # Find initial orientation angle that gives smallest total difference from propagation direction on circular domain (use cosine function to take into account periodic boundaries)
            for j in range(0, length - 1):

                # Consider average orientation angle between times as propagation direction is calculated using the displacement between times
                thetaAv += math.cos(direction[j] - ((theta[j + 1] + theta[j]) / 2))
                theta2Av += math.cos(direction[j] - ((theta2[j + 1] + theta2[j]) / 2))

            # Find average quantity for easy comparison
            thetaAv /= (length - 1)
            theta2Av /= (length - 1)



        # Else set initial orientation angle as closest to average direction of propagation in processive region(s):
        else:

            # Define lists for angle comparison (in this case do not want to consider all points, just those where object moves processively, so list is easier than array)
            propDirection = list()
            thetaDirection = list()
            theta2Direction = list()

            # For a track with multiple processive regions need to repeat analysis for each region, flag == 0 implies next b == 1 point is the first of a new processive region
            first = 0

            for j in range(0, length - 1):

                # If point on track is processive
                if b[j] == 1:

                    # If this is the first point of a processive region
                    if first == 0:

                        # Set initial propagation direction
                        propDirection.append(phi[j])

                        # Set orientation angle corresponding to same time (consider average orientation angle between times as propagation direction is calculated using the displacement between times)
                        thetaDirection.append((theta[j + 1] + theta[j]) / 2)
                        theta2Direction.append((theta2[j + 1] + theta2[j]) / 2)

                        # Change flag as average propagation angles do not need to be found for other points in the processive region
                        first = 1

                    # If this is not the first point on the processive region
                    else:

                        # Find previous propagation angle for efficiency
                        prev = propDirection[len(propDirection) - 1]

                        # Assume that propagation angle cannot change by more than pi in a single timestep (also due to equivalence of angles on circular domain)
                        direction = phi[j]
                        if abs(direction - prev) > math.pi:
                            while abs(direction - prev) > math.pi:
                                if direction > prev:
                                    direction = direction - (2 * math.pi)
                                else:
                                    direction = direction + (2 * math.pi)

                        # Add new propagation direction to array
                        propDirection.append(direction)

                        # Set orientation angle corresponding to same time
                        thetaDirection.append((theta[j + 1] + theta[j]) / 2)
                        theta2Direction.append((theta2[j + 1] + theta2[j]) / 2)

                if j > 0:

                    # If processive region ends reset flag so that propagation direction at first point of next processive region can be calculated using averaging again
                    if b[j] == 0 and b[j - 1] == 1:
                        first = 0  # Reset flag

            # Calculate average quantities to compare two orientation angle trajectories to propagation angle trajectory (as before):
            thetaAv = 0
            theta2Av = 0

            # In this case only consider processive points
            for j in range(0, len(propDirection)):
                thetaAv += math.cos(propDirection[j] - thetaDirection[j])
                theta2Av += math.cos(propDirection[j] - theta2Direction[j])
            thetaAv /= len(propDirection)
            theta2Av /= len(propDirection)



        # DEBUG: Plot and output extra angular data for a single track
        # if length > 50:
        # if sum(b) > 0:
        #     print(b)
        #     x = np.linspace(1, length, length)
        #     xp = np.linspace(1, length - 1, length - 1)
        #     y = np.zeros(length)
        #     for i in range(0, length):
        #         y[i] = self.points[i][2]
        #     plt.plot(xp, phi)
        #     plt.plot(x, theta)
        #     plt.plot(x, theta2)
        #     plt.plot(x, y)
        #     plt.show()
        #     f = open("%sExtra_Angle_Data.txt" % fileDataOut, "w")
        #     for i in range(0, length - 1):
        #         f.write('{:.4f} {:.4f} {:.4f} {:.4f}\n'.format(self.points[i][2], theta[i], theta2[i], phi[i]))
        #     f.close()



        # Choose initial orientation / orientation trajectory based on which has a smaller average difference from the propagation angle using the cosine function (cos(0) = 1, so smaller difference generates larger value)
        # thetaBefore = theta[0]
        if theta2Av > thetaAv:

            for j in range(0, length):
                theta[j] = theta2[j]
        # thetaAfter = theta[0]
        # DEBUG: Check if initial orientation changes from orientation provided by tracking
        # if thetaBefore != thetaAfter:
        #     print('Changed: {:.4f} -> {:.4f}'.format(thetaBefore, thetaAfter))

        # Store all chosen orientation angles and distribution initialised to theta[0] = 0 (expect the distribution of these angles (for many objects) to evolve as a Gaussian distribution in time for a diffusive system)
        for j in range(0, length):
            ang.append((j, theta[j]))
            ang2.append((j, theta[j] - theta[0]))



        # Find difference between orientation angle and direction of propagation for all points on track:
        for i in range(0, length - 1):

            # Find average angle between initial and final positions to compare to propagation direction
            angAv = (theta[i + 1] + theta[i]) / 2

            # Exploit (2 * pi) symmetry of cosine function and (0 < angle < pi) domain of arccos function
            # (such that delta can be calculated in one line without requiring the (delta mod (2 * pi)) and (delta reflected through angle = 0 to the (0 < angle < pi) domain) steps)
            delta = math.acos(math.cos(phi[i] - angAv))  # Can just use phi here as any rotations by +/- (2 * pi) will not change delta

            # Store all changes in angles
            delAng.append((i, delta))



        # For radial histogram plotting define arrays containing lengths of each bar:
        # Define array of length of bars for delAng
        radiiDelAng = np.zeros(180)

        # Loop over all angles for histogram
        for angle in delAng:

            # For each angle at each time increment the size of the corresponding histogram bar by one (set index equal to angle in degrees)
            radiiDelAng[int(math.floor(angle[1] * (180 / math.pi)))] += 1
            # DEBUG: Print array index
            # print(int(math.floor(angle[1] * (180 / math.pi))))

        # Define array of length of bars for ang
        radiiAng = np.zeros((maxLen, 360))

        i = 0
        for angle in ang:

            # For radial histogram plotting need to ensure all data points are within the observable domain (-pi < angle < pi)
            angle_buff = angle[1]
            while angle_buff >= math.pi or angle_buff < -math.pi:
                if angle_buff >= math.pi:
                    angle_buff = angle_buff - (2 * math.pi)
                elif angle_buff < -math.pi:
                    angle_buff = angle_buff + (2 * math.pi)

            # Store angles as a function of dwell time, as expect angular distribution to decay over time
            radiiAng[i, int(math.floor(angle_buff * (180 / math.pi))) + 180] += 1
            i += 1

        # Define array of length of bars for ang2 (ang initialised to ang[0,1] = 0)
        radiiAng2 = np.zeros((maxLen, 360))

        i = 0
        for angle in ang2:

            angle_buff = angle[1]
            while angle_buff >= math.pi or angle_buff < -math.pi:
                if angle_buff >= math.pi:
                    angle_buff = angle_buff - (2 * math.pi)
                elif angle_buff < -math.pi:
                    angle_buff = angle_buff + (2 * math.pi)

            radiiAng2[i, int(math.floor(angle_buff * (180 / math.pi))) + 180] += 1
            i += 1



        # Find radial histogram arrays for delAng contributions associated with processive or diffusive regions of track
        delAngProc = list()
        delAngDiff = list()

        for i in range(0, length - 1):

            # As delAng requires knowledge of the current and next orientations and propagation directions, require values of b for both these points to be 1 to be defined as a processive contribution
            if b[i] == 1 and b[i + 1] == 1:
                delAngProc.append(delAng[i][1])
            elif b[i] == 0 and b[i + 1] == 0:
                delAngDiff.append(delAng[i][1])

        # Store processive region(s) results in same form as used previously
        radiiDelAngProc = np.zeros(180)

        # If there are no processive points, cannot add any points to corresponding array
        if len(delAngProc) > 0:
            for angle in delAngProc:
                radiiDelAngProc[int(math.floor(angle * (180 / math.pi)))] += 1
                # DEBUG: Print array index
                # print(int(math.floor(angle * (180 / math.pi))))

        # Repeat for diffusive region(s) of track
        radiiDelAngDiff = np.zeros(180)
        if len(delAngDiff) > 0:
            for angle in delAngDiff:
                radiiDelAngDiff[int(math.floor(angle * (180 / math.pi)))] += 1
                # DEBUG: Print array index
                # print(int(math.floor(angle * (180 / math.pi))))



        # Return all arrays containing angles and all arrays required for histogram plotting
        return (delAng, radiiDelAng, ang, radiiAng, ang2, radiiAng2, radiiDelAngProc, radiiDelAngDiff)



    def displacements(self, dt, dx, graining, ang, b):
        # Find displacement of each object on track for all times

        length = len(self.points)

        # Define arrays to store displacements, cumulative sum of displacements, and components of displacements parallel and perpendicular to orientation of object
        disp     = np.zeros(length - 1)
        dispCumu = np.zeros(length)
        dispComp = np.zeros((length - 1, 2))

        i = 0
        for point in self.points:

            # Can only define displacement with two points, so cannot calculate a displacement for the first point on the track
            if i > 0:

                # Find displacement between timesteps
                dispx = point[0] - buff[0]
                dispy = point[1] - buff[1]
                disp[i - 1] = np.sqrt((dispx * dispx) + (dispy * dispy)) * dx  # Convert from px to um (real-space units)

                # Add displacement to cumulative sum of displacements
                dispCumu[i] = dispCumu[i - 1] + disp[i - 1]

                # Store components of displacement parallel and perpendicular to orientation
                dispComp[i - 1, 0] = abs(disp[i - 1] * math.cos(ang[i - 1][1]))
                dispComp[i - 1, 1] = abs(disp[i - 1] * math.sin(ang[i - 1][1]))

            # Store previous point in buffer
            buff = point
            i += 1


        # Bin displacement data for histogram plotting
        dispBin = np.zeros(100)

        for i in range(0, length - 1):

            # Define the index for each displacement using a constant graining factor
            index = int(math.floor(disp[i] * (graining / dx)))

            # Limit index according to pre-selected size of array for plotting
            if index < 100:

                # Increment array for each occurrence of displacement value
                dispBin[index] += 1


        # Store total arc length as a function of time for processive region(s) of track
        dispProc = list()

        # Count number of processive regions on track
        numProc = 0

        # Only need to consider processive arc lengths if the track has a processive region
        if sum(b) > 0:

            # Initialise counter to store length of processive region
            procLen = 0

            # DEBUG: Check processive points on track
            # print(b)

            # Loop over all possible points on track
            for i in range(0, length):

                if b[i] == 1:

                    procLen += 1
                    # DEBUG: Check indexes being used to store data
                    # print('Proc: {}'.format(i))

                if i > 0:

                    # Check if a processive region has ended in array of points on track
                    if ((b[i - 1] == 1) and (b[i] == 0)) or ((i == length - 1) and (b[i - 1] == 1) and (b[i] == 1)):

                        # Define indexes that label the start and end of the processive region in the array storing all displacements for the track
                        if i == length - 1 and b[i] == 1:
                            k = i - procLen + 1
                            m = i + 1
                        else:
                            k = i - procLen
                            m = i

                        # DEBUG: Check limits of indexes for processive region
                        # print('length = {}, procLen = {}, i = {}'.format(length, procLen, i))
                        # print('Limits: {}, {}'.format(k, m))

                        # Define array to store displacements for processive region of track
                        disp_buff = np.zeros((procLen - 1, 2))

                        # Loop over all possible displacements between points along processive region of track
                        for j in range(1, procLen):

                            # Loop over points on processive region of track that have at least j points until end of region
                            for n in range(k, m - j):

                                # Count number of contributing elements for each value of 'j', and cumulative displacement
                                disp_buff[j - 1, 0] += 1  # Store at j - 1, so 0 index corresponds to the displacement over one timestep
                                disp_buff[j - 1, 1] += dispCumu[n + j] - dispCumu[n]
                                # DEBUG: Check indexes being used to store data
                                # print('Stored: {} - {} ({} -> {}, jump size: {})'.format(n + j, n, k, m, j))

                        # Store all displacement data in relevant array
                        dispProc.append(disp_buff)

                        # Reset buffer storing length of prcoessive region
                        procLen = 0

                        # Increment counter for number of processive regions on track
                        numProc += 1

                        # DEBUG: Print results for processive region
                        # print('Region ended: {}'.format(numProc))
                        # print(disp_buff)

        return (dispBin, dispComp, dispProc, numProc)



    def processivePlot(self, dataOutProc, fileDataOut, dt, dx, qdt1, qdt2, qs, b, ang, delAng):
        # Plot processive and diffusive regions of track with vector arrows indicating orientation

        length = len(self.points)

        # Define time track is observed for
        time = np.linspace(self.start * dt, (self.start + length - 1) * dt, length)

        # Plot diagnostics (qdt in both directions, qds, b):
        # Define plot axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # Plot qdt in both directions in time
        ax1.plot(time, qdt1, 'k-', label='q(t) forwards')
        ax1.plot(time, qdt2, 'b-', label='q(t) backwards')

        # Draw line of (constant) qs to show threshold for defining region as processive
        qsPlot = np.zeros(length)
        qsPlot.fill(qs)
        ax1.plot(time, qsPlot, 'c-', label='qs threshold')

        # Plot values of binary array to show when a region has been defined as processive on a separate scale for clarity
        ax2.plot(time, b, 'r-', label='b')

        ax1.set_xlabel('$t$ ($s$)')
        ax1.set_ylabel('$q(t)$', color='k')
        ax2.set_ylabel('$b(t)$', color='r')
        ax1.legend(loc=0)
        plt.show()



        # DEBUG: Plot delAng (difference between orientation angle and propagation direction) and displacement from initial position as a function of time, and output data
        time = np.linspace((self.start + 0.5) * dt, (self.start + 0.5 + length - 2) * dt, (length - 1))
        dela = np.zeros(length - 1)
        for i in range(0, length - 1):
            dela[i] = delAng[i][1]
        plt.plot(time, dela)
        plt.xlabel('Time ($s$)')
        plt.ylabel('Difference in Angle, $\\Phi$')
        plt.show()
        if dataOutProc == 1:
            f = open("%sTOI_delAng_Data.txt" % fileDataOut, "w")
            for i in range(0, length - 1):
                f.write('{:.4f} {:.4f}\n'.format(time[i], dela[i]))
            f.close()

        time = np.linspace(self.start * dt, (self.start + length - 1) * dt, length)
        delx = np.zeros(length)
        for i in range(0, length):
            delx[i] = np.sqrt(((self.points[i][0] - self.points[0][0]) ** 2) + ((self.points[i][1] - self.points[0][1]) ** 2)) * dx
        plt.plot(time, delx)
        plt.xlabel('Time ($s$)')
        plt.ylabel('Displacement from Initial Position ($\\mu m$)')
        plt.show()
        if dataOutProc == 1:
            f = open("%sTOI_delX_Data.txt" % fileDataOut, "w")
            for i in range(0, length):
                f.write('{:.4f} {:.4f}\n'.format(time[i], delx[i]))
            f.close()



        # Use binary array to plot diffusive and processive regions of tracks separately:
        # Add all points to a track array
        track = list()
        for i in range(0, length):
            track.append((self.points[i][0] * dx, self.points[i][1] * dx))

        # Plot track array to show regions between processive and diffusive regions of track
        plt.plot(*zip(*track))

        # Define lists to store track points for processive and diffusive regions separately
        procTrack = list()
        diffTrack = list()

        # Define flags for start of processive and diffusive regions of track
        firstProc = 1
        firstDiff = 1

        # Loop over length of track
        for i in range(0, length):

            # Use binary array to store track points in the correct list (processive or diffusive)
            if b[i] == 1:
                procTrack.append((self.points[i][0] * dx, self.points[i][1] * dx))
            else:
                diffTrack.append((self.points[i][0] * dx, self.points[i][1] * dx))

            # Processive or diffusive region cannot end until at least the second timestep
            if i > 0:

                # Check if a processive region has ended in array of points on track
                if ((b[i - 1] == 1) and (b[i] == 0)) or ((i == length - 1) and (b[i - 1] == 1) and (b[i] == 1)):

                    # Create array to store track points for plotting
                    procTrackPlot = np.zeros((len(procTrack), 2))

                    # Loop over points in processive region of track
                    j = 0
                    for point in procTrack:

                        procTrackPlot[j, 0] = point[0]
                        procTrackPlot[j, 1] = point[1]
                        j += 1

                    # For first processive region add label to legend
                    if firstProc == 1:

                        plt.plot(procTrackPlot[:, 0], procTrackPlot[:, 1], 'r', label='Processive')

                        # Change flag so only one label is added to legend per track
                        firstProc = 0

                    # Else can just plot point
                    else:

                        plt.plot(procTrackPlot[:, 0], procTrackPlot[:, 1], 'r')

                    # Re-initialise list storing processive track region so regions separated in time are not joined in plot
                    procTrack = list()


                # Check if a diffusive region has ended in array of points on track, and repeat analysis
                if ((b[i - 1] == 0) and (b[i] == 1)) or ((i == length - 1) and (b[i - 1] == 0) and (b[i] == 0)):

                    diffTrackPlot = np.zeros((len(diffTrack), 2))

                    j = 0
                    for point in diffTrack:
                        diffTrackPlot[j, 0] = point[0]
                        diffTrackPlot[j, 1] = point[1]
                        j += 1

                    if firstDiff == 1:
                        plt.plot(diffTrackPlot[:, 0], diffTrackPlot[:, 1], 'g', label='Diffusive')
                        firstDiff = 0
                    else:
                        plt.plot(diffTrackPlot[:, 0], diffTrackPlot[:, 1], 'g')

                    diffTrack = list()


        # Plot first point of track as a star so overall direction / displacement can be observed
        plt.plot(track[0][0], track[0][1], 'k*', markersize=12)


        # Draw unit vector arrows indicating orientation at all times
        for i in range(0, length):
            plt.arrow(track[i][0], track[i][1], math.cos(ang[i][1]) * dx, math.sin(ang[i][1]) * dx, head_width=0.5 * dx, head_length=1.0 * dx)


        # Create plot
        plt.xlabel('$x$ ($\\mu m$)')
        plt.ylabel('$y$ ($\\mu m$)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc=0)
        plt.show()


        # Output important data to file for track of interest
        if dataOutProc == 1:
            f = open("%sb_Data.txt" % fileDataOut, "w")
            for i in range(0, length):
                f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(time[i], qdt1[i], qdt2[i], qs, b[i]))
            f.close()

            f = open("%sTOI_Data.txt" % fileDataOut, "w")
            for i in range(0, length):
                f.write('{:.4f} {:.4f} {:.4f} {:.4f}\n'.format(time[i], self.points[i][0] * dx, self.points[i][1] * dx, ang[i][1]))
            f.close()



    def MSD(self, b, ang):
        # Calculate MSD of object

        length = len(self.points)

        # Initialise arrays to store MSD variables
        MSD     = np.zeros((length - 1, 3))
        MSDProc = list()
        MSDDiff = list()


        # Loop over all possible displacements between points along track
        for i in range(1, length):

            # Loop over points on track that have at least i points until end of track
            for j in range(0, length - i):

                # Count number of contributing elements for each value of 'i', and cumulative MSD
                MSD[i - 1, 0] += 1  # Index 0 corresponds to displacement over 1 timestep
                MSD[i - 1, 1] += (((self.points[j + i][0] - self.points[j][0]) ** 2) + ((self.points[j + i][1] - self.points[j][1]) ** 2))
                MSD[i - 1, 2] += ((ang[j + i][1] - ang[j][1]) ** 2)


        # Repeat analysis to calculate the MSD for processive and diffusive regions of the track:
        # Define lists to store track points for processive and diffusive regions separately
        procTrack = list()
        diffTrack = list()

        # Loop over length of track
        for i in range(0, length):

            # Use binary array to store track points in the correct list (processive or diffusive)
            if b[i] == 1:
                procTrack.append((self.points[i][0], self.points[i][1]))
                # DEBUG: Check indexes being used to store data
                # print('Proc: {}'.format(i))
            else:
                diffTrack.append((self.points[i][0], self.points[i][1]))
                # DEBUG: Check indexes being used to store data
                # if sum(b) > 0:
                #     print('Diff: {}'.format(i))

            # Processive or diffusive region cannot end until at least the second timestep
            if i > 0:

                # Check if a processive region has ended in array of points on track
                if ((b[i - 1] == 1) and (b[i] == 0)) or ((i == length - 1) and (b[i - 1] == 1) and (b[i] == 1)):

                    # Create array to store track points for calculating MSD
                    procTrackArr = np.zeros((len(procTrack), 3))

                    # Create array to store MSD for track region
                    MSD_buff = np.zeros((len(procTrack) - 1, 3))

                    # Loop over points in processive region of track
                    j = 0
                    # Store orientation of object at each point (to extract from angle array require separate index)
                    if i == length - 1 and b[i] == 1:
                        k = i - len(procTrack) + 1
                    else:
                        k = i - len(procTrack)

                    # Loop over all processive points in region
                    for point in procTrack:

                        # DEBUG: Check indexes being used to store data
                        # print('Stored: {}, {} (length of procTrack: {})'.format(j, k, len(procTrack)))
                        # print('(Stored, corresponding): ({:.4f}, {:.4f}), ({:.4f}, {:.4f})'.format(point[0], self.points[k][0], point[1], self.points[k][1]))

                        # Store positions and angles for points
                        procTrackArr[j, 0] = point[0]
                        procTrackArr[j, 1] = point[1]
                        procTrackArr[j, 2] = ang[k][1]

                        # Increment counter variables
                        j += 1
                        k += 1

                    # Calculate MSD for processive track region:
                    # Loop over all possible displacements between points along processive region of track
                    for j in range(1, len(procTrack)):

                        # Loop over points on processive region of track that have at least j points until end of region
                        for k in range(0, len(procTrack) - j):

                            # Count number of contributing elements for each value of 'j', and cumulative MSD
                            MSD_buff[j - 1, 0] += 1  # Index 0 corresponds to displacement over 1 timestep
                            MSD_buff[j - 1, 1] += (((procTrackArr[k + j, 0] - procTrackArr[k, 0]) ** 2) + ((procTrackArr[k + j, 1] - procTrackArr[k, 1]) ** 2))
                            MSD_buff[j - 1, 2] += ((procTrackArr[k + j, 2] - procTrackArr[k, 2]) ** 2)

                    # Store MSD for processive track region in list
                    MSDProc.append(MSD_buff)

                    # Re-initialise list storing processive track region so regions separated in time are not joined in plot
                    procTrack = list()


                # Check if a diffusive region has ended in array of points on track
                if ((b[i - 1] == 0) and (b[i] == 1)) or ((i == length - 1) and (b[i - 1] == 0) and (b[i] == 0)):

                    # Repeat same analysis as for a processive region, but this time for diffusive region:
                    diffTrackArr = np.zeros((len(diffTrack), 3))
                    MSD_buff = np.zeros((len(diffTrack) - 1, 3))

                    j = 0
                    if i == length - 1 and b[i] == 0:
                        k = i - len(diffTrack) + 1
                    else:
                        k = i - len(diffTrack)

                    for point in diffTrack:

                        # DEBUG: Check indexes being used to store data
                        # if sum(b) > 0:
                        #     print('Stored: {}, {} (length of diffTrack: {})'.format(j, k, len(diffTrack)))
                        #     print('(Stored, corresponding): ({:.4f}, {:.4f}), ({:.4f}, {:.4f})'.format(point[0], self.points[k][0], point[1], self.points[k][1]))

                        diffTrackArr[j, 0] = point[0]
                        diffTrackArr[j, 1] = point[1]
                        diffTrackArr[j, 2] = ang[k][1]
                        j += 1
                        k += 1

                    # Calculate MSD for processive track region:
                    # Loop over all possible displacements between points along processive region of track
                    for j in range(1, len(diffTrack)):

                        # Loop over points on processive region of track that have at least j points until end of region
                        for k in range(0, len(diffTrack) - j):

                            # Count number of contributing elements for each value of 'j', and cumulative MSD
                            MSD_buff[j - 1, 0] += 1  # Index 0 corresponds to displacement over 1 timestep
                            MSD_buff[j - 1, 1] += (((diffTrackArr[k + j, 0] - diffTrackArr[k, 0]) ** 2) + ((diffTrackArr[k + j, 1] - diffTrackArr[k, 1]) ** 2))
                            MSD_buff[j - 1, 2] += ((diffTrackArr[k + j, 2] - diffTrackArr[k, 2]) ** 2)

                    # Store MSD for processive track region in list
                    MSDDiff.append(MSD_buff)

                    diffTrack = list()

        # Units of MSD converted to physical units (px -> um) in plotting function for simplicity


        return (MSD, MSDProc, MSDDiff)
