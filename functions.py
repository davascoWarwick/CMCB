import numpy as np
import sep
import fitsio
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from track import Track
import scipy
from scipy.stats import norm
from scipy.optimize import curve_fit



def countNumBound(dataOutNumBound, fileDataOut, numImages, frameInit, dt, tracks):
    # Count number of objects observed as a function of time

    # Define array to store number of bound objects
    numBound = np.zeros(numImages)

    # Loop over all objects observed
    N = 0
    for track in tracks:
        N += 1
        # Increment array counting number of bound objects for each time the object was observed
        for i in range(track.start - frameInit, track.start + track.trackLength() - frameInit):
            numBound[i] += 1

    print('Average number of objects observed per frame ({:.4f} +/- {:.4f})'.format(np.mean(numBound), np.std(numBound)))

    # Plot number of objects observed each timestep
    time = np.linspace(frameInit, (frameInit + numImages - 1), numImages) * dt
    plt.plot(time, numBound, 'k-')
    plt.xlabel('Time, $t$ ($s$)')
    plt.ylabel('Number of Bound Objects')
    plt.show()


    # Can output total number of observed objects each timestep to file
    if dataOutNumBound == 1:
        f = open("%sNumber_Bound_Data.txt" % fileDataOut, "w")
        for i in range(frameInit, frameInit + numImages):
            f.write('{} {}\n'.format(i, numBound[i - frameInit]))
        f.close()



def ExpGaussianFit(x, x0, sig, lam, C1, C2):
    # Create an exponentially-modified Gaussian distribution with average x0, standard deviation sig, time constant 1/lam, normalisation constant C1, and constant offset C2

    return C1 * (lam / 2) * np.exp((lam / 2) * ((2 * x0) + (lam * (sig ** 2)) - (2 * x))) * scipy.special.erfc((x0 + (lam * (sig ** 2)) - x) / (np.sqrt(2) * sig)) + C2



def areasPlot(dataOutAreas, fileDataOut, graphLog, dx, graining, tracks):
    # Count number of objects of each area size to find characteristic shape sizes in images

    # Calculate new graining multiplier for area distributions to take into account factor of 36 in area (6 in each ellipse parameter) calculation
    newGraining = 10

    # Find the maximum and minimum areas of a detected particle
    maxArea = 0
    minArea = 999
    for track in tracks:
        a = np.array(track.readAlla())
        b = np.array(track.readAllb())
        for i in range(0, track.trackLength()):
            if math.pi * a[i] * b[i] > maxArea:
                maxArea = math.pi * a[i] * b[i]
            if math.pi * a[i] * b[i] < minArea:
                minArea = math.pi * a[i] * b[i]
    print('Max. area of a detection = {}, Min. area of a detection = {}'.format(maxArea, minArea))

    # Convert maximum area to an index limit for binning
    lim = math.floor(maxArea)
    newLim = math.floor(maxArea / newGraining)

    # Define array to store number of particles of each area size
    area = np.zeros((newLim + 1, 2))
    area[:, 0] = [(m + 0.5) * (dx * dx) * newGraining for m in range(newLim + 1)]  # Convert pixel units to um
    dA = area[1, 0] - area[0, 0]

    # Fill distribution array with all areas of particle detections
    for track in tracks:
        a = np.array(track.readAlla())
        b = np.array(track.readAllb())
        for i in range(0, track.trackLength()):
            area[math.floor((math.pi * a[i] * b[i]) / newGraining), 1] += 1
    
    # DEBUG: Print number of detections in each area bin
    # for i in range(0, len(area)):
    #     print(area[i, :])
    
    # Normalise distribution (taking into account bin size) to find PDF of area
    area[:, 1] /= (sum(area[:, 1]) * dA)

    # Plot area distribution
    plt.plot(area[:, 0], area[:, 1], 'kD')
    plt.xlabel('Area of Detection ($\\mu m^2$)')
    plt.ylabel('Probability Distribution Function $P(A)$ ($\\mu m^{-2}$)')

    # Check normalisation of PDF:
    print('Normalisation of area PDF = {:.4f}'.format(sum(area[:, 1] * dA)))

    # Calculate fit to curve (give function estimate of parameters p0 for accurate and efficient fitting)
    ExpGaussian_fit, cov = curve_fit(ExpGaussianFit, area[:, 0], area[:, 1], p0=[area[np.argmax(area[:, 1]), 0], 1.0, 1.0, 1.0, 0.0])
    print("Fit to PDF(A):")
    print(ExpGaussian_fit)
    print(cov)
    print("\n")

    # Plot fit (use finer grid than binning as have no problem of small number of data points per bin for analytical distribution)
    x = np.array([(m + 0.5) * ((dx * dx) / graining) for m in range((graining * lim) + 1)])
    y = ExpGaussianFit(x, *ExpGaussian_fit)
    plt.plot(x, y, 'r:', linewidth=2, label='ExpGaussian fit')

    # Add line representing the peak area for the distribution
    peakArea = x[np.argmax(y)]
    print('Peak area = {}'.format(peakArea))
    avArea = ExpGaussian_fit[0] + (1 / ExpGaussian_fit[2])
    print('Average area = {} (Gaussian peak = {})'.format(avArea, ExpGaussian_fit[0]))
    plt.axvline(x=peakArea, color='b', linestyle='dashed', linewidth=2)
    plt.axvline(x=avArea, color='g', linestyle='dashed', linewidth=2)
    plt.axvline(x=ExpGaussian_fit[0], color='c', linestyle='dashed', linewidth=2)
    plt.show()


    # Find log of curve
    if graphLog == 1:

        # Find the log of the number of occurrences of each area and the fitted curve
        logArea = list()
        for i in range(0, lim + 1):
            # Make sure not to log zero entries of data array
            if area[i, 0] != 0 and area[i, 1] != 0:
                logArea.append((math.log(area[i, 0], math.e), math.log(area[i, 1], math.e)))
        logxy = list()
        for i in range(0, (lim * graining) + 1):
            if x[i] != 0 and y[i] > 0:  # Fit y-value can drop below 0
                logxy.append((math.log(x[i], math.e), math.log(y[i], math.e)))

        # Plot log of displacement data
        plt.plot(*zip(*logArea), 'kD')
        plt.plot(*zip(*logxy), 'r:', linewidth=2, label='ExpGaussian fit')
        plt.axvline(x=math.log(peakArea, math.e), color='b', linestyle='dashed', linewidth=2)
        plt.xlabel('log(Area of Detection ($\\mu m^2$))')
        plt.ylabel('log(Probability Distribution Function $P(A)$ ($\\mu m^{-2}$))')
        plt.show()


    # Can output area data to file
    if dataOutAreas == 1:
        f = open("%sArea_Data.txt" % fileDataOut, "w")
        for i in range(0, newLim + 1):
            f.write('{:.4f} {:.4f}\n'.format(area[i, 0], area[i, 1]))
        f.close()


    return peakArea



def MSDPlot(dataOutMSD, fileDataOut, dt, dx, MSD):
    # Plot and fit MSD and MSAD curves


    # Convert MSD data array to physical units (columns 0 (count) and 2 (MSAD) are already in physical units (number and rads))
    MSD[:, 1] *= (dx ** 2)


    # Find length of data array
    length = len(MSD[:, 0])

    # Find suitable time-axis limit
    maxLen_buff = length
    for i in range(0, length):
        # Store time when there are no objects still bound long enough to contribute to array
        if MSD[i, 0] == 0:
            maxLen_buff = i - 1
            break

    # Set maximum time-axis limit as 10s (50 timesteps) to display data with minimum errors
    maxLen = min(maxLen_buff * dt, 20)

    # Define time corresponding to data
    time = np.linspace(dt, length * dt, length)


    # Plot MSD data
    plt.plot(time, MSD[:, 1], 'kD', label='Raw Data')

    # Only fit early-time data to find processive or diffusive behaviour
    cutoff1 = 0
    cutoff2 = int(maxLen / (4 * dt))  # Fit over a quarter of the observable region

    # Fit MSD data with quadratic function:
    if cutoff2 > 5:  # Arbitrary cutoff for reasonable fit with 3 degrees of freedom

        # Define arrays to store early-time MSD data
        time_buff = np.zeros(cutoff2 - cutoff1)
        MSD_buff  = np.zeros(cutoff2 - cutoff1)
        MSAD_buff = np.zeros(cutoff2 - cutoff1)
        k = 0
        for j in range(cutoff1, cutoff2):
            time_buff[k] = time[j]
            MSD_buff[k]  = MSD[j, 1]
            MSAD_buff[k] = MSD[j, 2]
            k += 1

        # Fit MSD data
        z, cov = np.polyfit(time_buff, MSD_buff, 2, cov=True)  # 2 is order of polynomial (quadratic)
        MSD_fit = np.poly1d(z)
        MSD_new = MSD_fit(time_buff)
        print("MSD fit:")
        print(MSD_fit)
        print(np.sqrt(np.diag(cov)))

        # Plot fit to MSD data
        plt.plot(time_buff, MSD_new, 'g-', label='Quadratic Fit')

    plt.xlabel('Time, $t$ ($s$)')
    plt.xlim([0, maxLen])
    plt.ylabel('MSD ($\\mu m^2$)')
    plt.legend(loc=2)
    plt.show()

    # Plot MSAD data
    plt.plot(time, MSD[:, 2], 'kD', label='Raw Data')

    # Fit MSAD data with quadratic function (in the same way):
    if cutoff2 > 5:

        z, cov = np.polyfit(time_buff, MSAD_buff, 2, cov=True)
        MSAD_fit = np.poly1d(z)
        MSAD_new = MSAD_fit(time_buff)
        print("MSAD fit:")
        print(MSAD_fit)
        print(np.sqrt(np.diag(cov)))

        plt.plot(time_buff, MSAD_new, 'g-', label='MSAD Fit')

    plt.xlabel('Time, $t$ ($s$)')
    plt.xlim([0, maxLen])
    plt.ylabel('MSAD')
    plt.legend(loc=2)
    plt.show()


    # Can output MSD and MSAD data to file
    if dataOutMSD == 1:
        f = open("%sMSD_Data.txt" % fileDataOut, "w")
        for i in range(0, length):
            f.write('{:.4f} {:.4f}\n'.format(time[i], MSD[i, 1]))
        f.close()

        f = open("%sMSAD_Data.txt" % fileDataOut, "w")
        for i in range(0, length):
            f.write('{:.4f} {:.4f}\n'.format(time[i], MSD[i, 2]))
        f.close()



def dispPlot(dataOutDisp, fileDataOut, graphLog, disp, dispTot):
    # Plot number of times an object (at any time or as part of any track) has a displacement of a given magnitude between timesteps

    # Plot displacement data as probability distribution
    dp = disp[1, 0] - disp[0, 0]
    plt.plot(disp[:, 0], disp[:, 1] / sum(disp[:, 1] * dp), 'kD')
    plt.xlabel('Displacement Magnitude ($\\mu m$)')  # Equivalent to instantaneous velocity between timesteps
    plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
    plt.ylabel('Probability Distribution Function, $P(r)$')
    plt.show()

    # DEBUG: Test if PDF is normalised
    print('Normalisation of r PDF = {:.4f}'.format(sum((disp[:, 1] / sum(disp[:, 1] * dp)) * dp)))


    # Find log of curve and apply linear fit (only interested in fitting linear behaviour)
    if graphLog == 1:

        # Find the log of the number of occurrences of each displacement
        logDisp = list()
        for i in range(0, len(disp)):
            # Make sure not to log zero entries of data array
            if disp[i, 1] != 0:
                logDisp.append((disp[i, 0], math.log(disp[i, 1] / sum(disp[:, 1] * dp), math.e)))  # Lin-log distribution

        # Plot log of displacement data
        plt.plot(*zip(*logDisp), 'kD', label='log data')

        # Find linear fit for central region of logged data:
        # Define buffer arays to store subset of displacement array data for fitting
        x_buff = np.zeros(math.floor(len(logDisp) / 2))
        y_buff = np.zeros(math.floor(len(logDisp) / 2))

        # Only interested in fitting central region of logged data
        for i in range(0, math.floor(len(logDisp) / 2)):
            x_buff[i] = logDisp[i + math.floor(len(logDisp) / 2)][0]
            y_buff[i] = logDisp[i + math.floor(len(logDisp) / 2)][1]

        # Fit disp data:
        z = np.polyfit(x_buff, y_buff, 1)  # Fit with polynomial of order 1 (linear fit)
        disp_fit = np.poly1d(z)
        disp_new = disp_fit(x_buff)
        print("log(Displacement Magnitude) fit:")
        print(disp_fit)

        # Plot fitted curve
        plt.plot(x_buff, disp_new, color='darkorange', label='Linear fit')
        plt.xlabel('Displacement Magnitude ($\\mu m$)')
        plt.ylabel('log(Probability Distribution Function, $P(r)$)')
        plt.show()


    # Can calculate the one-dimensional probability distribution (probability of a displacement per unit area)
    prob1D = np.zeros((len(disp), 2))

    # Calculate the bin width of the possible displacements
    dp = disp[1, 0] - disp[0, 0]

    # Loop over all possible displacement magnitudes
    for i in range(0, len(disp)):

        # Fill first column with the possible values of the magnitude of the displacements (for plotting)
        prob1D[i, 0] = disp[i, 0]

        # Fill second column with (normalised) probability distribution for displacement per unit area
        prob1D[i, 1] = disp[i, 1] / (sum(disp[:, 1]) * 2 * math.pi * dp * disp[i, 0])

    # DEBUG: Test if PDF is normalised
    print('Normalisation of (r, theta) PDF: {:.4f}'.format(sum(prob1D[:, 1] * prob1D[:, 0] * dp * 2 * math.pi)))

    # Plot probability distribution function
    plt.plot(prob1D[:, 0], prob1D[:, 1], 'kD')
    plt.xlabel('Displacement Magnitude ($\\mu m$)')  # Equivalent to instantaneous velocity between timesteps
    plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
    plt.ylabel('Probability Distribution Function, $P(r, \\theta)$ ($\\mu m^{-2}$)')
    plt.show()


    # Can output displacement data to file
    if dataOutDisp == 1:
        f = open("%sDisplacement_Step_Data.txt" % fileDataOut, "w")
        for i in range(0, len(disp)):
            f.write('{:.4f} {:.4f}\n'.format(disp[i, 0], disp[i, 1]))
        f.close()

        # DEBUG: Output average straight-line displacement and variables required to calculate its uncertainty
        f = open("%sAverage_Displacement_Data.txt" % fileDataOut, "w")
        f.write('{} {} {}\n'.format(np.mean(dispTot), np.std(dispTot), len(dispTot)))
        f.close()



def GaussianFit(x, x0, sig, C1, C2):
    # Create a Gaussian distribution with average x0, standard deviation sig, normalisation constant C1, and constant offset C2

    return C1 * np.exp(-((x - x0) ** 2) / (2 * (sig ** 2))) + C2



def radPlot360(dataOutOrientations, fileDataOut, graphAllThetaAllt, dt, maxLen, radii):
    # Plot orientation angle data as radial histogram

    # Define array of possible angles for histogram axis
    theta = [(m + 0.5 - 180) * (math.pi / 180) for m in range(0, 360)]

    # Can plot time series of orientation angle
    if graphAllThetaAllt == 1:

        # Normalise and find maximum for distribution of orientations after a single timestep for visibility when plotting
        radMax = (radii[1, :] / sum(radii[1, :])).max(axis=0)

        # Loop over maximum length of track
        for i in range(0, maxLen):

            # Normalise radial histogram data for current time
            radiiNorm = radii[i, :] / sum(radii[i, :])

            # Plot radial histogram for current time
            plt.plot(theta, radiiNorm, '-', label='Raw data')

            # Apply a Gaussian fit to the data:
            # Calculate average orientation angle
            radiiAv = sum(theta[:] * radiiNorm[:]) / len(theta)

            # Calculate standard deviation of data
            sig = np.sqrt(sum(radiiNorm[:] * ((theta[:] - radiiAv) ** 2) / len(theta)))

            # Calculate fit (give function estimate of parameters p0 for accurate and efficient fitting)
            Gaussian_fit, cov = curve_fit(GaussianFit, theta[:], radiiNorm, p0=[radiiAv, sig, 1.0, 0.0])
            print('Fit at time: {:.4f}'.format(i * dt))
            print(Gaussian_fit)
            print("\n")

            # Plot fit
            plt.plot(theta[:], GaussianFit(theta[:], *Gaussian_fit), 'r:', linewidth=4, label='Gaussian fit')
            plt.xlabel('Angle (rad)')
            plt.ylabel('Probability')
            plt.ylim(0, max(max(radiiNorm), radMax))
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()


    # Set plot dimensions
    fig = plt.figure(figsize=(10., 10.))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Have already defined array of length of histogram bars (radii array using number of occurrences of each angle)
    # Define array of width of bars
    width = math.pi / 180

    # Define array to store total number of occurences of each angle for all times (sum over time index)
    radii_buff = np.zeros(360)
    for i in range(0, maxLen):
        radii_buff[:] += radii[i, :]

    # Normalise distribution
    radii_norm = radii_buff / sum(radii_buff)

    # Define bars for histogram
    bars = ax.bar(theta, radii_norm, width=width, bottom=0.0)
    for r, bar in zip(radii_norm, bars):
        bar.set_facecolor(cm.jet(r / max(radii_norm)))
        bar.set_alpha(1.0)

    # Edit plot variables
    ax.set_thetamin(-180)
    ax.set_thetamax(180)
    plt.xlabel('Probability', labelpad=20)
    plt.show()


    # Can output orientation angle data to file
    if dataOutOrientations == 1:
        f = open("%sAngle_Data.txt" % fileDataOut, "w")
        for i in range(0, 360):
            f.write('{:.4f} {:.4f}\n'.format(theta[i], radii_buff[i]))
        f.close()



def radPlot180(dataOutDelAng, fileDataOut, radii):
    # Plot magnitude of difference between orientation angle and direction of propagation data as radial histogram

    # Set plot dimensions
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Define array of possible angles for histogram axis
    theta = [(m + 0.5) * (math.pi / 180) for m in range(0, 180)]

    # Have already defined array of length of histogram bars (radii array using number of occurrences of each angle)
    # Define array of width of bars
    width = math.pi / 180

    # Normalise distribution
    radii_norm = radii / sum(radii)

    # Define bars for histogram
    bars = ax.bar(theta, radii_norm, width=width, bottom=0.0)
    for r, bar in zip(radii_norm, bars):
        bar.set_facecolor(cm.jet(r / max(radii_norm)))
        bar.set_alpha(1.0)

    # Edit plot variables
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_yticklabels([])
    plt.xlabel('Probability', labelpad=20)
    plt.show()


    # Can output angle change data to file
    if dataOutDelAng == 1:
        f = open("%sAngle_Change_Data.txt" % fileDataOut, "w")
        for i in range(0, len(theta)):
            f.write('{:.4f} {:.4f}\n'.format(theta[i], radii[i]))
        f.close()



def radPlot90(radii):
    # Plot magnitude of minimum difference between orientation angle and direction of propagation data as radial histogram assuming axial, elliptical symmetry (orientation is symmetric under rotation of pi radians)

    # Set plot dimensions
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Define array of possible angles for histogram axis
    theta = [(m + 0.5) * (math.pi / 180) for m in range(0, 90)]

    # Have already defined array of length of histogram bars (radii array using number of occurrences of each angle)
    # Define array of width of bars
    width = math.pi / 180

    # Normalise distribution
    radii_norm = radii / sum(radii)

    # Define bars for histogram
    bars = ax.bar(theta, radii_norm, width=width, bottom=0.0)
    for r, bar in zip(radii_norm, bars):
        bar.set_facecolor(cm.jet(r / max(radii_norm)))
        bar.set_alpha(1.0)

    # Edit plot variables
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    plt.xlabel('Probability', labelpad=20)
    plt.show()



def freqPlot(dataOutTret, dataOutXret, fileDataOut, graphLog, dt, dx, graining, maxLen, maxDisp, N1Frame, procCount, b, bestTracks):
    # Plot number of occurrences of dwell time and displacement events

    # Plot number of occurrences of each value of dwell time:
    # Define time corresponding to data
    time = [m * dt for m in range(2, maxLen + 1)]  # Assume minimum possible dwell time for observation (if bundle observed for two frames, minimum possible dwell time is dt, and maximum is 2 * dt)

    # Define array to calculate average retention time
    tret = []
    # Define arrays to calculate average retention times only for tracks with (or without) processive regions
    tretProc = []
    tretDiff = []

    # Define array to store number of occurrences of each value of dwell time
    trackDistT = np.zeros((maxLen - 1, 2))

    # Set first column in array equal to dwell time of object, and second column equal to the number of occurrences of that dwell time
    trackDistT[:, 0] = time


    # DEBUG: Check that no single frame detections are present in analysis
    for track in bestTracks:
        if track.trackLength() == 1:
            print('Single frame detections included in analysis')


    # Loop over all tracks
    count = 0
    maxProcDwell = 0
    maxDiffDwell = 0
    for track in bestTracks:

        # Assume dwell time = trackLength (in time) - 1, the number of timesteps observed by the detection
        trackDistT[track.trackLength() - 2, 1] += 1  # Index 0 corresponds to dwell time of 1 timestep (dt)
        tret.append(track.trackLength() * dt)

        # Store dwell times only for tracks with (or without) processive regions
        if sum(b[count]) != 0:
            tretProc.append(track.trackLength() * dt)
            if track.trackLength() * dt > maxProcDwell:
                maxProcDwell = track.trackLength() * dt
        else:
            tretDiff.append(track.trackLength() * dt)
            if track.trackLength() * dt > maxDiffDwell:
                maxDiffDwell = track.trackLength() * dt
        count += 1

        # DEBUG: Check indexing for dwell time arrays
        # print('{} -> {} : 1 = {}, 2 = {}'.format(track.trackLength(), track.trackLength() - 2, trackDistT[track.trackLength() - 2, 0], track.trackLength() * dt))


    # Calculate and output average dwell time with error
    print('Average dwell time = ({:.4} +/- {:.4})s'.format(np.mean(tret), np.std(tret)))  # Large error in dwell time due to split in timescales, and need to take into account 1/sqrt(N)
    if procCount > 0:
        print('Average processive track dwell time = ({:.4} +/- {:.4})s, maximum = {:.4}s'.format(np.mean(tretProc), np.std(tretProc), maxProcDwell))  # Large error in dwell time due to split in timescales, and need to take into account 1/sqrt(N)
    print('Average diffusive track  dwell time = ({:.4} +/- {:.4})s, maximum = {:.4}s'.format(np.mean(tretDiff), np.std(tretDiff), maxDiffDwell))  # Large error in dwell time due to split in timescales, and need to take into account 1/sqrt(N)

    # Plot dwell time data
    plt.plot(trackDistT[:, 0], trackDistT[:, 1] / (sum(trackDistT[:, 1]) * dt), 'kD')
    plt.xlabel('Time spent bound, $t_b$ ($s$)')
    plt.xlim([0, 5])
    plt.ylabel('Probability Distribution Function, $P(t_b)$ ($s^{-1}$)')
    plt.show()

    # DEBUG: Test if PDF is normalised
    print('Normalisation of t_b PDF = {:.4f}'.format(sum((trackDistT[:, 1] / (sum(trackDistT[:, 1]) * dt)) * dt)))

    # Plot a lin-log plot to observe characteristic timescales
    if graphLog == 1:

        # Find the log of the number of occurrences of each dwell time
        logTrackDistT = list()
        for i in range(0, maxLen - 1):
            # Make sure not to log zero entries of data array
            if trackDistT[i, 1] != 0:
                logTrackDistT.append((trackDistT[i, 0], math.log(trackDistT[i, 1] / (sum(trackDistT[:, 1]) * dt), math.e)))

        # Plot log of dwell time data
        plt.plot(*zip(*logTrackDistT), 'kD')
        plt.xlabel('Time spent bound, $t_b$ ($s$)')
        plt.ylabel('log(Probability Distribution Function, $P(t_b)$ ($s^{-1}$))')
        plt.show()



    # Can output dwell time data to file
    if dataOutTret == 1:
        f = open("%sRetention_Times_Data.txt" % fileDataOut, "w")
        # First line of output is number of single frame detections
        f.write('{:.4f} {:.4f}\n'.format(0.2, N1Frame))
        # Following lines are all other dwell time detections
        for i in range(0, maxLen - 1):
            f.write('{:.4f} {:.4f}\n'.format(trackDistT[i, 0], trackDistT[i, 1]))
        f.close()

        # DEBUG: Output average dwell time and variables required to calculate its uncertainty
        f = open("%sAverage_Retention_Time_Data.txt" % fileDataOut, "w")
        f.write('{} {} {}\n'.format(np.mean(tret), np.std(tret), len(tret)))
        f.close()

        # DEBUG: Output dwell times for each individual track separately
        # f = open("%sRetention_Times_All_Data.txt" % fileDataOut, "w")
        # for track in bestTracks:
        #     f.write('{:.4f}\n'.format((track.trackLength() - 1) * dt))
        # f.close()



    # Plot number of occurrences of each value of straight-line displacement:
    # Use graining to better parametrise displacement distribution (need +1 here as particles can have zero net displacement)
    lim = math.floor(maxDisp * graining)
    trackDistDSL = np.zeros((lim + 1, 2))
    trackDistDSL[:, 0] = [(m + 0.5) * (dx / graining) for m in range(lim + 1)]
    for track in bestTracks:
        # displacement = track.dispSL() * dx
        # index = displacement * (graining / dx) = track.dispSL() * graining, as function returns displacement in pixel units already
        trackDistDSL[(math.floor(track.dispSL() * graining)), 1] += 1
    dp = trackDistDSL[1, 0] - trackDistDSL[0, 0]  # Bin width for finding probabilities
    plt.plot(trackDistDSL[:, 0], trackDistDSL[:, 1] / (sum(trackDistDSL[:, 1]) * dp), 'kD')
    plt.xlabel('Total displacement from initial position ($\\mu m$)')
    plt.xlim([0, 0.6])  # Arbitrary limit to observe interesting behaviour
    plt.ylabel('Probability Distribution Function, $P(r_sl)$ ($\\mu m^{-1}$)')
    plt.show()

    # DEBUG: Test if PDF is normalised
    print('Normalisation of r_sl PDF = {:.4f}'.format(sum((trackDistDSL[:, 1] / (sum(trackDistDSL[:, 1]) * dp)) * dp)))

    # DEBUG: Find average total displacement from PDF
    print('Average total displacement = {:.4f}um'.format(sum(trackDistDSL[:, 0] * (trackDistDSL[:, 1] / (sum(trackDistDSL[:, 1]) * dp)) * dp)))

    # Plot a lin-log plot to observe characteristic length-scales
    if graphLog == 1:
        logTrackDistDSL = list()
        for i in range(0, lim + 1):
            if trackDistDSL[i, 1] != 0:
                logTrackDistDSL.append((trackDistDSL[i, 0], math.log(trackDistDSL[i, 1] / (sum(trackDistDSL[:, 1]) * dp), math.e)))
        plt.plot(*zip(*logTrackDistDSL), 'kD')
        plt.xlabel('Total displacement from initial position ($\\mu m$)')
        plt.ylabel('log(Probability)')
        plt.show()

     # Can calculate the one-dimensional probability distribution (probability of a displacement per unit area), as used for the displacement analysis previously
    prob1D = np.zeros((lim + 1, 2))

    # Calculate the bin width of the possible displacements
    dp = trackDistDSL[1, 0] - trackDistDSL[0, 0]

    # Neglect displacements that round to zero, as these would be eliminated for small enough bin sizes to get the smooth top of a Gaussian (instead of NaN)
    for i in range(0, lim + 1):

        # Fill first column with the possible values of the magnitude of the displacements (for plotting)
        prob1D[i, 0] = trackDistDSL[i, 0]

        # Fill second column with (normalised) probability distribution for displacement per unit area
        prob1D[i, 1] = trackDistDSL[i, 1] / (sum(trackDistDSL[:, 1]) * 2 * math.pi * dp * trackDistDSL[i, 0])

    # DEBUG: Test if new PDF is normalised:
    print('Normalisation of (r_sl, theta) PDF = {:.4f}'.format(sum(prob1D[:, 1] * prob1D[:, 0] * dp * 2 * math.pi)))

    # Plot probability distribution
    plt.plot(prob1D[:, 0], prob1D[:, 1], 'kD')
    plt.xlabel('Total displacement from initial position ($\\mu m$)')
    plt.xlim([0, 0.3])  # Different arbitrary limit to that used previously
    plt.ylabel('Probability Distribution Function, $P(r_sl, \\theta)$ ($\\mu m^{-2}$)')
    plt.show()



    # Can output straight-line displacement data to file
    if dataOutXret == 1:
        f = open("%sDisplacement_Total_Data.txt" % fileDataOut, "w")
        for i in range(0, lim + 1):
            f.write('{:.4f} {:.4f}\n'.format(trackDistDSL[i, 0], trackDistDSL[i, 1]))
        f.close()



    # Plot number of occurrences of each value of cumulative displacement (path arc length):
    # Find maximum cumulative displacement for all tracks
    maxDispCumu = 0
    for track in bestTracks:
        if track.dispCumu() > maxDispCumu:
            maxDispCumu = track.dispCumu()

    # Repeat previous displacement analysis
    lim = math.floor(maxDispCumu * graining)
    trackDistDC = np.zeros((lim + 1, 2))
    trackDistDC[:, 0] = [(m + 0.5) * (dx / graining) for m in range(lim + 1)]
    for track in bestTracks:
        # displacement = track.dispCumu() * dx
        # index = displacement * (graining / dx) = track.dispCumu() * graining, as function returns displacement in pixel units already
        trackDistDC[(math.floor(track.dispCumu() * graining)), 1] += 1
    dp = trackDistDC[1, 0] - trackDistDC[0, 0]  # Bin width for finding probabilities
    plt.plot(trackDistDC[:, 0], trackDistDC[:, 1] / (sum(trackDistDC[:, 1]) * dp), 'kD')
    plt.xlabel('Total arc length of path ($\\mu m$)')
    plt.xlim([0, 1])
    plt.ylabel('Probability Distribution Function, $P(r_c)$ ($\\mu m^{-1}$)')
    plt.show()

    # DEBUG: Test if PDF is normalised
    print('Normalisation of r_c PDF = {:.4f}'.format(sum((trackDistDC[:, 1] / (sum(trackDistDC[:, 1]) * dp)) * dp)))

    # DEBUG: Find average cumulative displacement from PDF
    print('Average cumulative displacement = {:.4f}um\n'.format(sum(trackDistDC[:, 0] * (trackDistDC[:, 1] / (sum(trackDistDC[:, 1]) * dp)) * dp)))

    # Plot a lin-log plot to observe characteristic displacement scales
    if graphLog == 1:
        logTrackDistDC = list()
        for i in range(0, lim + 1):
            if trackDistDC[i, 1] != 0:
                logTrackDistDC.append((trackDistDC[i, 0], math.log(trackDistDC[i, 1] / sum(trackDistDC[:, 1]), math.e)))
        plt.plot(*zip(*logTrackDistDC), 'kD')
        plt.xlabel('Total arc length of path ($\\mu m$)')
        plt.ylabel('log(Probability)')
        plt.show()



    # Can output cumulative displacement data to file
    if dataOutXret == 1:
        f = open("%sDisplacement_Cumulative_Data.txt" % fileDataOut, "w")
        for i in range(0, lim + 1):
            f.write('{:.4f} {:.4f}\n'.format(trackDistDC[i, 0], trackDistDC[i, 1]))
        f.close()



def procVarPlot(dataOutProc, fileDataOut, dt, bestTrackCount, b, numProc, dispProc):
    # Plot variables related to total dwell times and displacements for each state (processive and diffusive) for tracks with a processive region

    # Define lists to store time spent moving processively, total dwell time, and the fraction of the total dwell time spent moving processively
    tauProc = list()
    tauTot  = list()
    tauFrac = list()

    # Define counter to store total number of tracks with a processive region
    j = 0

    # Loop over all tracks being analysed
    for i in range(0, bestTrackCount):

        # Only consider tracks with a processive region
        if sum(b[i]) != 0:

            # Store relevant times in lists
            tauProc.append((sum(b[i]) - numProc[i]) * dt)  # Time spent moving processively
            tauTot.append((len(b[i]) - 1) * dt)  # Total dwell time
            tauFrac.append((sum(b[i]) - numProc[i]) / (len(b[i]) - 1))  # Fraction of total dwell time spent moving processively

            j += 1

    procTracks = j
    print('Out of {} tracks with (at least one) processive region: average dwell time = ({:4f} +/- {:.4f}), average time spent moving processively = ({:.4f} +/- {:.4f})'.format(j, np.mean(tauTot), np.std(tauTot), np.mean(tauProc), np.std(tauProc)))

    # Store data in single array for efficiency
    tau_all = np.zeros((j, 2))
    tau_all[:, 0] = tauTot[:]
    tau_all[:, 1] = tauProc[:]

    # Plot time spent moving processively against total time spent bound
    plt.plot(tau_all[:, 0], tau_all[:, 1], 'kD')

    # Plot line of x = y to represent maximum value of tauProc for a given value of tauTot
    lim = int(round(max(tauProc[:]) / dt) + 1)
    xy = np.zeros((lim, 2))
    xy[:, 0] = np.linspace(dt, lim * dt, lim)
    xy[:, 1] = xy[:, 0]
    plt.plot(xy[:, 0], xy[:, 1], 'r-')

    plt.xlabel("Time spent bound, $t_b$ ($s$)")
    plt.ylabel("Time spent moving processively, $t_p$ ($s$)")
    plt.ylim([min(tauProc[:]) - dt, max(tauProc[:]) + dt])
    plt.show()

    # DEBUG: Plot fraction of time spent moving processively against total time spent bound, and plot contours of constant numbers of timesteps spent moving not processively
    plt.plot(tau_all[:, 0], tauFrac[:], 'kD')
    lim = int(round(max(tauTot[:]) / dt) + 1)
    t = np.linspace(dt, lim * dt, lim)
    for i in range(0, lim):
        y = np.zeros(lim)
        for j in range(0, lim):
            y[j] = (t[j] - (i * dt)) / t[j]
        plt.plot(t, y, 'r-')
    plt.xlabel("Time spent bound, $t_b$ ($s$)")
    plt.xlim([min(tauTot[:]) - dt, max(tauTot[:]) + dt])
    plt.ylabel("Fraction of time spent moving processively when bound, $t_p/t_b$")
    plt.ylim([min(tauFrac[:]) * 0.9, max(tauFrac[:]) + (min(tauFrac[:]) * 0.1)])
    plt.show()

    # DEBUG: Plot fraction of time spent moving processively as a boxplot
    plt.boxplot(tauFrac)
    plt.ylabel("Fraction of time spent moving processively when bound, $t_p/t_b$")
    plt.xticks([1], [''])
    plt.show()



    # Can output times spent moving processively to file
    if dataOutProc == 1:
        f = open("%sProcessive_Times_Data.txt" % fileDataOut, "w")
        for i in range(0, len(tau_all)):
            f.write('{:.4f} {:.4f}\n'.format(tau_all[i, 0], tau_all[i, 1]))
        f.close()



    # Plot cumulative displacement (path arc length) during processive regions of track against the time spent moving processively (expect linear plot with gradient = intrinsic velocity)
    plt.plot(dispProc[:, 0], dispProc[:, 1], 'kD', label='Raw Data')

    # Only fit early-time data to find averaged behaviour (later data points have fewer contributing tracks)
    cutoff1 = 0
    cutoff2 = int(len(dispProc[:, 0]) / 2)  # Fit over a half of the observable region

    # Only fit data if there are enough data points to define a reasonable fit
    if cutoff2 > 5:  # Arbitrary cutoff for reasonable fit with 3 degrees of freedom

        # Define arrays to store early-time cumulative processive displacement data
        time_buff = np.zeros(cutoff2 - cutoff1)
        dispProc_buff = np.zeros(cutoff2 - cutoff1)
        j = 0
        for i in range(cutoff1, cutoff2):
            time_buff[j] = dispProc[i, 0]
            dispProc_buff[j] = dispProc[i, 1]
            j += 1

        # Fit cumulative processive displacement data
        z, cov = np.polyfit(time_buff, dispProc_buff, 1, cov=True)  # 1 is order of polynomial (linear)
        dispProc_fit = np.poly1d(z)
        dispProc_new = dispProc_fit(time_buff)
        print("dispProc fit:")
        print(dispProc_fit)

        # Plot fit to data
        plt.plot(time_buff, dispProc_new, 'g-', label='Linear Fit')

    plt.xlabel('Time spent moving processively, $t_p$ ($s$)')
    plt.ylabel('Total arc length of path ($\\mu m$)')
    plt.legend(loc=2)
    plt.show()

    # DEBUG: Plot cumulative displacement while moving processively as a boxplot
    plt.boxplot(dispProc[:, 1])
    plt.ylabel('Total arc length of path ($\\mu m$)')
    plt.xticks([1], [''])
    plt.show()



    # Can output cumulative displacements during time spent moving processively
    if dataOutProc == 1:
        f = open("%sProcessive_Displacement_Cumulative_Data.txt" % fileDataOut, "w")
        for i in range(0, len(dispProc)):
            f.write('{:.4f} {:.4f}\n'.format(dispProc[i, 0], dispProc[i, 1]))
        f.close()

        # DEBUG: Output average bound / processive velocity and variables required to calculate its uncertainty
        f = open("%sAverage_Processive_Velocity_Data.txt" % fileDataOut, "w")
        f.write('{} {} {}\n'.format(z[0], cov[0, 0], procTracks))
        f.close()



def procPlot(dataOutProc, fileDataOut, graphAllProc, dx, graining, procCount, b, dispComp, disp, bestTracks):
    # Plot displacement variables for processive and diffusive regions of track separately

    # Define arrays for storing data
    dispProc = list()
    dispProcDist = np.zeros(100)
    dispProcPara = list()
    dispProcParaDist = np.zeros(100)
    dispProcPerp = list()
    dispProcPerpDist = np.zeros(100)

    dispDiff = list()
    dispDiffDist = np.zeros(100)
    dispDiffPara = list()
    dispDiffParaDist = np.zeros(100)
    dispDiffPerp = list()
    dispDiffPerpDist = np.zeros(100)

    # Define counter to store track number
    j = 0

    # Loop over all tracks
    for track in bestTracks:

        # If track has processive region store displacement variables
        if sum(b[j]) != 0:

            # DEBUG: Print b array
            # print('NEW:')
            # print(b[j])
            # print(b[j][0])
            # print(b[j][1])
            # print(len(b[j]))

            # DEBUG: Check dispComp indexing
            # print(dispComp[j])
            # print(len(dispComp[j]))

            # For each point on the track store the parallel and perpendicular components of the displacement relative to the object orientation for processive and diffusive regions
            for i in range(0, len(dispComp[j])):

                # DEBUG: Check dispComp indexing
                # print('i = {}'.format(i))
                # print(dispComp[j][i, 0])
                # print(dispComp[j][i, 1])

                # If displacement is during processive region
                if b[j][i] == 1 and b[j][i + 1] == 1:

                    # print('Processive displacement')

                    dispProc.append(np.sqrt((dispComp[j][i, 0] ** 2) + (dispComp[j][i, 1] ** 2)))
                    index = int(math.floor(np.sqrt((dispComp[j][i, 0] ** 2) + (dispComp[j][i, 1] ** 2)) * (graining / dx)))
                    if index < 100:
                        dispProcDist[index] += 1

                    dispProcPara.append(dispComp[j][i, 0])
                    index = int(math.floor(dispComp[j][i, 0] * (graining / dx)))
                    if index < 100:
                        dispProcParaDist[index] += 1

                    dispProcPerp.append(dispComp[j][i, 1])
                    index = int(math.floor(dispComp[j][i, 1] * (graining / dx)))
                    if index < 100:
                        dispProcPerpDist[index] += 1

                # If displacement is during diffusive region
                elif b[j][i] == 0 and b[j][i + 1] == 0:

                    # print('Diffusive displacement')

                    dispDiff.append(np.sqrt((dispComp[j][i, 0] ** 2) + (dispComp[j][i, 1] ** 2)))
                    index = int(math.floor(np.sqrt((dispComp[j][i, 0] ** 2) + (dispComp[j][i, 1] ** 2)) * (graining / dx)))
                    if index < 100:
                        dispDiffDist[index] += 1

                    dispDiffPara.append(dispComp[j][i, 0])
                    index = int(math.floor(dispComp[j][i, 0] * (graining / dx)))
                    if index < 100:
                        dispDiffParaDist[index] += 1

                    dispDiffPerp.append(dispComp[j][i, 1])
                    index = int(math.floor(dispComp[j][i, 1] * (graining / dx)))
                    if index < 100:
                        dispDiffPerpDist[index] += 1

        # Increment counter for labelling loop / track
        j += 1


    print('Number of tracks with processive regions   = {} / {} = {:.4f}'.format(procCount, j, procCount / j))

    # Calculate average displacements, and plot and output results if there is at least one detected track with a processive region
    if procCount > 0:

        dispProc_av = np.mean(dispProc)
        dispProc_std = np.std(dispProc)

        dispProcPara_av = np.mean(dispProcPara)
        dispProcPara_std = np.std(dispProcPara)

        dispProcPerp_av = np.mean(dispProcPerp)
        dispProcPerp_std = np.std(dispProcPerp)

        dispDiff_av = np.mean(dispDiff)
        dispDiff_std = np.std(dispDiff)

        dispDiffPara_av = np.mean(dispDiffPara)
        dispDiffPara_std = np.std(dispDiffPara)

        dispDiffPerp_av = np.mean(dispDiffPerp)
        dispDiffPerp_std = np.std(dispDiffPerp)

        print('Average processive displacement magnitude  = ({:.4f} +/- {:.4f})um'.format(dispProc_av, dispProc_std))
        print('Average processive parallel component      = ({:.4f} +/- {:.4f})um'.format(dispProcPara_av, dispProcPara_std))
        print('Average processive perpendicular component = ({:.4f} +/- {:.4f})um'.format(dispProcPerp_av, dispProcPerp_std))

        print('Average diffusive displacement magnitude   = ({:.4f} +/- {:.4f})um'.format(dispDiff_av, dispDiff_std))
        print('Average diffusive parallel component       = ({:.4f} +/- {:.4f})um'.format(dispDiffPara_av, dispDiffPara_std))
        print('Average diffusive perpendicular component  = ({:.4f} +/- {:.4f})um'.format(dispDiffPerp_av, dispDiffPerp_std))

        # Plot displacement results:
        if graphAllProc == 1:

            # Plot displacement data as probability distribution
            dp = disp[1, 0] - disp[0, 0]
            plt.plot(disp[:, 0], dispProcDist[:] / sum(dispProcDist[:] * dp), 'r-', label='Processive')
            plt.plot(disp[:, 0], dispDiffDist[:] / sum(dispDiffDist[:] * dp), 'g-', label='Diffusive')

            # DEBUG: Test if PDF is normalised
            print('Normalisation of r(proc) PDF = {:.4f}'.format(sum((dispProcDist[:] / sum(dispProcDist[:] * dp)) * dp)))
            print('Normalisation of r(diff) PDF = {:.4f}'.format(sum((dispDiffDist[:] / sum(dispDiffDist[:] * dp)) * dp)))

            # Add lines representing the average displacements for both components
            plt.axvline(x=dispProc_av, color='r', linestyle='dashed')
            plt.axvline(x=dispDiff_av, color='g', linestyle='dashed')

            plt.xlabel('Displacement Magnitude ($\\mu m$)')
            plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
            plt.ylabel('Probability Distribution Function, $P(r_{proc/diff})$')
            plt.legend()
            plt.show()

            # Can calculate the one-dimensional probability distribution (probability of a displacement per unit area)
            prob1DProc = np.zeros((len(disp), 2))
            prob1DDiff = np.zeros((len(disp), 2))

            # Calculate the bin width of the possible displacements
            dp = disp[1, 0] - disp[0, 0]

            # Loop over all possible displacement magnitudes
            for i in range(0, len(disp)):

                # Fill first column with the possible values of the magnitude of the displacements (for plotting)
                prob1DProc[i, 0] = disp[i, 0]
                prob1DDiff[i, 0] = disp[i, 0]

                # Fill second column with (normalised) probability distribution for displacement per unit area
                prob1DProc[i, 1] = dispProcDist[i] / (sum(dispProcDist[:]) * 2 * math.pi * dp * disp[i, 0])
                prob1DDiff[i, 1] = dispDiffDist[i] / (sum(dispDiffDist[:]) * 2 * math.pi * dp * disp[i, 0])

            # DEBUG: Test if new PDF is normalised
            print('Normalisation of (r_proc, theta) PDF = {:.4f}'.format(sum(prob1DProc[:, 1] * prob1DProc[:, 0] * dp * 2 * math.pi)))
            print('Normalisation of (r_diff, theta) PDF = {:.4f}'.format(sum(prob1DDiff[:, 1] * prob1DDiff[:, 0] * dp * 2 * math.pi)))

            # Plot probability distribution function
            plt.plot(prob1DProc[:, 0], prob1DProc[:, 1], 'rD', label='Processive')
            plt.plot(prob1DDiff[:, 0], prob1DDiff[:, 1], 'gD', label='Diffusive')
            plt.xlabel('Displacement Magnitude ($\\mu m$)')
            plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
            plt.ylabel('Probability Distribution Function, $P(r_{proc/para}, \\theta)$ ($\\mu m^{-2}$)')
            plt.show()



            # Plot the probability distributions of the components of the displacements parallel and perpendicular to the orientation of the object for processive and diffusive regions
            dp = disp[1, 0] - disp[0, 0]

            plt.plot(disp[:, 0], dispProcParaDist / sum(dispProcParaDist * dp), 'r-', label='Processive, Parallel')
            plt.plot(disp[:, 0], dispProcPerpDist / sum(dispProcPerpDist * dp), color='tomato', label='Processive, Perpendicular')

            plt.plot(disp[:, 0], dispDiffParaDist / sum(dispDiffParaDist * dp), 'g-', label='Diffusive, Parallel')
            plt.plot(disp[:, 0], dispDiffPerpDist / sum(dispDiffPerpDist * dp), color='lime', label='Diffusive, Perpendicular')

            # DEBUG: Test if PDF is normalised
            print('Normalisation of r(proc, para) PDF = {:.4f}'.format(sum((dispProcParaDist / sum(dispProcParaDist * dp)) * dp)))
            print('Normalisation of r(proc, perp) PDF = {:.4f}'.format(sum((dispProcPerpDist / sum(dispProcPerpDist * dp)) * dp)))

            print('Normalisation of r(diff, para) PDF = {:.4f}'.format(sum((dispDiffParaDist / sum(dispDiffParaDist * dp)) * dp)))
            print('Normalisation of r(diff, perp) PDF = {:.4f}'.format(sum((dispDiffPerpDist / sum(dispDiffPerpDist * dp)) * dp)))

            plt.xlabel('Displacement Magnitude ($\\mu m$)')
            plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
            plt.ylabel('Probability Distribution Function $P(r_{proc/diff, para/perp})$')
            plt.legend()
            plt.show()



        # Can output distributions of the displacements (and their components parallel and perpendicular to the orientation of the object) for processive and diffusive regions to file
        if dataOutProc == 1:
            f = open("%sProcessive_Displacements_Data.txt" % fileDataOut, "w")
            for i in range(0, len(disp)):
                f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(disp[i, 0], dispProcDist[i], dispProcParaDist[i], dispProcPerpDist[i], dispDiffDist[i], dispDiffParaDist[i], dispDiffPerpDist[i]))
            f.close()



def dispComponentsPlot(dataOutComp, fileDataOut, graphLog, dx, graining, dispComp, disp, bestTracks):
    # Plot components of the displacement parallel and perpendicular to the orientation of the object for all objects

    # Define arrays for storing data
    dispPara = list()
    dispPara_dist = np.zeros(100)
    dispPerp = list()
    dispPerp_dist = np.zeros(100)

    # Loop over all tracks
    j = 0
    for track in bestTracks:

        # DEBUG: Check dispComp indexing
        # print('NEW:')
        # print(dispComp[j])
        # print(len(dispComp[j]))

        # For each point on the track store the parallel and perpendicular components of the displacement relative to the object orientation
        for i in range(0, len(dispComp[j])):

            # DEBUG: Check dispComp indexing
            # print('i = {}'.format(i))
            # print(dispComp[j][i, 0])
            # print(dispComp[j][i, 1])

            # Add parallel component of the displacement to list
            dispPara.append(dispComp[j][i, 0])

            # Store each value of component of the displacement as a distribution using an index proportional to the displacement
            index = int(math.floor(dispComp[j][i, 0] * (graining / dx)))
            if index < 100:
                dispPara_dist[index] += 1

            # Repeat for the perpendicular component
            dispPerp.append(dispComp[j][i, 1])

            index = int(math.floor(dispComp[j][i, 1] * (graining / dx)))
            if index < 100:
                dispPerp_dist[index] += 1

        # Increment counter for labelling loop
        j += 1

    # Calculate average displacements per timestep in processive and diffusive regions
    dispPara_av = np.mean(dispPara)
    dispPara_std = np.std(dispPara)
    dispPerp_av = np.mean(dispPerp)
    dispPerp_std = np.std(dispPerp)

    # Print average displacement results
    print('Number of tracks                                            = {}'.format(j))
    print('Average displacement component parallel to orientation      = ({:.4f} +/- {:.4f})$\\mu m$'.format(dispPara_av, dispPara_std))
    print('Average displacement component perpendicular to orientation = ({:.4f} +/- {:.4f})$\\mu m$\n'.format(dispPerp_av, dispPerp_std))


    # Plot the probability distributions of the components of the displacements parallel and perpendicular to the orientation of the object
    dp = disp[1, 0] - disp[0, 0]
    plt.plot(disp[:, 0], dispPara_dist / sum(dispPara_dist * dp), 'm-', label='Parallel')
    plt.plot(disp[:, 0], dispPerp_dist / sum(dispPerp_dist * dp), 'y-', label='Perpendicular')

    # DEBUG: Test if PDF is normalised
    print('Normalisation of r(para) PDF = {:.4f}'.format(sum((dispPara_dist / sum(dispPara_dist * dp)) * dp)))
    print('Normalisation of r(perp) PDF = {:.4f}'.format(sum((dispPerp_dist / sum(dispPerp_dist * dp)) * dp)))

    # Add lines representing the average displacements for both components
    plt.axvline(x=dispPara_av, color='m', linestyle='dashed')
    plt.axvline(x=dispPerp_av, color='y', linestyle='dashed')

    plt.xlabel('Displacement ($\\mu m$)')
    plt.xlim([0, 0.0318 * 6])  # Set to the maximum allowed separation between moves on a track (dx * maxSep)
    plt.ylabel('Probability Distribution Function $P(r_{para/perp})$')
    plt.legend()
    plt.show()


    # Plot a lin-log plot to observe characteristic timescales
    if graphLog == 1:

        # Find the log of the number of occurrences of each component of the displacement
        logDispPara_dist = list()
        logDispPerp_dist = list()
        for i in range(0, len(dispPara_dist)):
            # Make sure not to log zero entries of data array
            if dispPara_dist[i] != 0:
                logDispPara_dist.append((disp[i, 0], math.log(dispPara_dist[i] / sum(dispPara_dist * dp), math.e)))
            if dispPerp_dist[i] != 0:
                logDispPerp_dist.append((disp[i, 0], math.log(dispPerp_dist[i] / sum(dispPerp_dist * dp), math.e)))

        # Plot log of displacement component data
        plt.plot(*zip(*logDispPara_dist), 'm-', label='Parallel')
        plt.plot(*zip(*logDispPerp_dist), 'y-', label='Perpendicular')

        # Fit displacement component data:
        # Create arrays to store central linear region of data
        logDispPara_buff = np.zeros((13, 2))
        logDispPerp_buff = np.zeros((13, 2))
        for i in range(0, 13):
            logDispPara_buff[i, :] = logDispPara_dist[i + 2][:]
            logDispPerp_buff[i, :] = logDispPerp_dist[i + 2][:]

        # Apply linear fits to arrays:
        # Fit parallel component data
        z = np.polyfit(logDispPara_buff[:, 0], logDispPara_buff[:, 1], 1)
        logDispPara_fit = np.poly1d(z)
        logDispPara_new = logDispPara_fit(logDispPara_buff[:, 0])
        print("Parallel component of displacement fit:")
        print(logDispPara_fit)
        print("\n")

        # Fit perpendicular component data
        z = np.polyfit(logDispPerp_buff[:, 0], logDispPerp_buff[:, 1], 1)
        logDispPerp_fit = np.poly1d(z)
        logDispPerp_new = logDispPerp_fit(logDispPerp_buff[:, 0])
        print("Perpendicular component of displacement fit:")
        print(logDispPerp_fit)
        print("\n")

        # Plot linear fits to displacement component data
        plt.plot(logDispPara_buff[:, 0], logDispPara_new[:], 'r-', label='Linear fit to parallel data')
        plt.plot(logDispPerp_buff[:, 0], logDispPerp_new[:], 'g-', label='Linear fit to perpendicular data')

        plt.xlabel('Displacement ($\\mu m$)')
        plt.ylabel('log(Probability Distribution Function $P(r_{para/perp})$)')
        plt.legend()
        plt.show()


    # Can output components of displacement parallel and perpendicular to the orientation of the object data to file
    if dataOutComp == 1:
        f = open("%sDisplacement_Components_Data.txt" % fileDataOut, "w")
        for i in range(0, len(disp)):
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(disp[i, 0], dispPara_dist[i], dispPerp_dist[i]))
        f.close()
