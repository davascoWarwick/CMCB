########################################################################################################################
# Code to perform single particle tracking analysis on acto-myosin iSCAT images, created by Lewis Mosby 2018-2019
#
# Input (either):
# - Time series of TIF images (taken using iSCAT) containing objects to be tracked and analysed
# - Set of data files containing two-dimensional positions, orientations, and shapes for each frame of a video (x pos., y pos., orientation angle, semi-major axis length, semi-minor axis length, time) of previously tracked objects to be analysed
#
# Outputs (any of):
# - Map of tracks observed over all times, or superimposed over each frame of video image
# - Map of individual track of interest (including orientation and parameters used to determine whether regions of track are processive or diffusive)
# - Number of bound objects observed each frame
# - Distribution of detected object areas
# - Mean squared displacement / mean squared angular displacement data (curves for all objects or for regions of processive and diffusive track separately)
# - Distributions of individual step, total (straight-line), and cumulative (path arc length) displacements (curves for all objects or for tracks with processive regions separately)
# - Distributions of orientation angles and the difference between an objects orientation and its direction of propagation (curve for all objects or for processive and diffusive regions of track separately)
# - Distribution of orientation as a function of time to show Gaussian evolution
# - Dwell time data (time spent bound or moving processively)
# - Distributions of displacements parallel and perpendicular to the orientation of the detected object
########################################################################################################################



# Begin code ###########################################################################################################

# Import all necessary libraries
import numpy as np
import sep
import fitsio
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse

# Import track class and plotting functions from files
from track import Track
from functions import countNumBound, areasPlot, MSDPlot, dispPlot, radPlot360, radPlot180, radPlot90, freqPlot, procVarPlot, procPlot, dispComponentsPlot

# All functions are used in this program as follows:
# countNumBound(dataOutNumBound, fileDataOut, numImages, frameInit, dt, tracks)
# areasPlot(dataOutAreas, fileDataOut, graphLog, dx, graining, tracks)
# proc = track.processiveDefine(dt, dx, diffusivity)
# proc_buff = track.processiveFind(dt, dx, diffusivity, minLen)
# radii_buff = track.angles(fileDataOut, maxLen, bTot[i])
# disp_buff = track.displacements(dt, dx, graining, delAng, bTot[i])
# track.processivePlot(dataOutProc, fileDataOut, dt, dx, qdt1, qdt2, qs, bTot[i], ang, delAng)
# MSD_buff = track.MSD(bTot[i], ang)
# MSDPlot(dataOutMSD, fileDataOut, dt, dx, MSD)
# dispPlot(dataOutDisp, fileDataOut, graphLog, disp, dispTot)
# radPlot360(dataOutOrientations, fileDataOut, 0, dt, maxLen, radiiAng)
# radPlot180(dataOutDelAng, fileDataOut, radiiDelAng)
# radPlot90(radiiBuff)
# freqPlot(dataOutTret, dataOutXret, fileDataOut, graphLog, dt, dx, graining, maxLen, maxDisp, bestTracks)
# procVarPlot(dataOutProc, fileDataOut, dt, bestTrackCount, bTot, numProc, dispProc)
# procPlot(dataOutProc, fileDataOut, graphAllProc, dx, graining, procCount, bTot, dispComp, disp, bestTracks)
# dispComponentsPlot(dataOutComp, fileDataOut, graphLog, dx, graining, dispComp, disp, bestTracks)



# Initialise variables #################################################################################################

# Set plot dimensions and font variables
rcParams['figure.figsize'] = [14., 10.]
plt.rcParams.update({'font.size': 18})

# Initialise lists of track variables:
ongoingTracks = list()                      # Stores particles still being tracked
completeTracks = list()                     # Stores particles whose tracks are completed
bestTracks = list()                         # Stores relevant (longest) tracks to be plotted / outputted
buff = []                                   # Buffer for storage / swapping

# Set video / image variables:
video = 6  # 5                                   # Choose video to analyse
frameInit = 1  # 352                             # First frame to be observed
frameFin = 750  # 651                              # Last frame to be observed
numImages = frameFin - frameInit + 1        # Number of frames in film to be observed (initial to final inclusive)
dataOut = 1                                 # Output track data as text files
dataIn = 0                                  # Read in track data as text files instead of finding tracks from videos
numFiles = 18857  # 14363                            # Number of files to read in

# Set tracking variables:
objErr = 1.5                                # Number of standard deviations above the background intensity an object has to be to be detected
minArea = 40                                # Minimum size of an object to still be detected
minSize = 149.3                             # Minimum size of an object to be included in analysis
maxSep = 6                                  # Maximum separation allowed between positions at different timesteps for object to be defined as on the same track
maxSizeSep = 0.8128473525 / (0.0318 ** 2)   # Maximum size difference between timesteps for object to be defined as on the same track (/ as the same object)
tOngoing = 0                                # Time allowed between detections of an object to still allow tracking (for tOngoing = 1 a track can be joined at times 0 and 2 even if there was no object detected as part of the track at t = 1 etc.)
minLenCutoff = 1                            # Minimum length for track data to be outputted
minDispCutoff = 0                           # Minimum displacement for track data to be outputted
maxLenCutoff = 1000                         # Maximum length for track data to be outputted (unless otherwise required set value >= length of simulation)
maxDispCutoff = 1000                        # Maximum displacement for track data to be outputted
minLen = 5                                  # Required length of processive region to be outputted
dx = 0.0318                                 # px     -> distance (micrometres, um) conversion
dt = 0.2                                    # frames -> time (seconds, s) conversion

# Define input / output file names:
fileImagesIn =       "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/FinalFilms/ImageOut%d/" % video
fileDataIn =         "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/FinalFilms/DataOut%d/" % video
fileTrackImagesOut = "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/FinalFilms/TrackImageOut%d/" % video
fileTrackDataOut =   "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/FinalFilms/DataOut%d/" % video
fileDataOut =        "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/FinalFilms/DataOut%d/OtherData/" % video
# fileImagesIn =       "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/NewFilms/ImageOut%d/" % video
# fileDataIn =         "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/NewFilms/DataOut%d/"  % video
# fileTrackImagesOut = "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/NewFilms/TrackImageOut%d/"  % video
# fileTrackDataOut =   "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/NewFilms/DataOut%d/"  % video
# fileDataOut =        "/home/phungd/Documents/PhD/ImageAnalysis/MyWork/NewFilms/DataOut%d/OtherData/"  % video

# Find diffusivity by applying linear fit to MSD for objects that have maximum track lengths of 20 (gradient = 2*D), has been converted to (um / s).
# Values change for each ATP concentration / video. Values for videos 5-11:
# 5:  0.01558 # 0.01106
# 6:  0.01604 # 0.005372
# 7:  0.02397
# 8:  -
# 9:  0.00898
# 10: 0.01178
# 11: 0.01408
diffusivity = 0.01106 / 2                   # Diffusivity of objects used for defining processive tracks

# Set flags for inputting / outputting data:
graphLog = 2                                # Turn on plotting of log curves for fitting
# Number of objects data:
graphNumBound = 1                           # Turn on plotting of the number of bound objects each timestep
dataOutNumBound = graphNumBound             # Output number of bound objects data
# Area of objects data
graphAllAreas = 1                           # Turn on plotting of the distribution of areas of detected objects
dataOutAreas = graphAllAreas                # Output area data
# Track data:
graph1Frame = 2                             # Turn on plotting of each frame of video with objects shown
graph1ImageTrack = 2                        # Turn on plotting of each frame of video with single track drawn (defined below)
TOI = 127  # 11788  # 38381                        # Track of interest for drawing of single track
graphAllFinal = 2                           # Turn on plotting of all tracks at the end of analysis, so distribution can be observed
# Frequency of events data:
graphAllFreq = 1                            # Turn on plotting of dwell time and displacement event frequencies
dataOutTret = graphAllFreq                  # Output dwell time data
dataOutXret = graphAllFreq                  # Output data for total displacements while bound
# Velocity / processive data:
graphAllDisp = 1                            # Turn on plotting of all particle displacements (instantaneous velocities)
dataOutDisp = graphAllDisp                  # Output data for individual displacement magnitudes
graphAllComp = 2                            # Turn on plotting for parallel / perpendicular velocity component distributions
dataOutComp = graphAllComp                  # Output data for velocities parallel and perpendicular to orientation
graphAllMSD = 2                             # Turn on MSD and MSAD plotting
dataOutMSD = graphAllMSD                    # Output MSD data
graph1Proc = 2                              # Turn on plotting of data for individual processive tracks
graphAllProc = 2                            # Turn on plotting for processive/diffusive velocity distributions
dataOutProc = graphAllProc                  # Output data for processive tracks (e.g.: time spent moving processively, displacements while moving processively, etc.)
# Angular data:
graphAllOrientations = 2                    # Turn on plotting of object orientation (Ang and Ang_0, domain -(pi / 2) < theta < (pi / 2))
dataOutOrientations = graphAllOrientations  # Output orientation data
graphAllOrientationsAllt = 2                # Turn on plotting of object orientation at each timestep
graphAlldelAng = 2                          # Turn on plotting of the difference between orientation and propagation angles
dataOutDelAng = graphAlldelAng              # Output data for difference between orientation and propagation angles



# Begin tracking #######################################################################################################

# Find tracks from videos:
if dataIn == 0:

    # Initialise variable to number each track for reference
    trackID = 1

    # Loop through all images in video
    for i in range(frameInit, frameFin + 1):

        # Read in fits image (create from video using ImageJ software)
        data = fitsio.read("%sImage%d.fits" % (fileImagesIn, i))
        data = data.astype(float)

        # Measure spatially varying background on image
        bkg = sep.Background(data)
        bkg_image = bkg.back()
        bkg_rms = bkg.rms()

        # Subtract background from image
        data_sub = data - bkg

        # Extract objects from image after background subtraction
        objects = sep.extract(data_sub, objErr, err=bkg.globalrms, minarea=minArea)
        print('Image {}, Num. objects: {}'.format(i, len(objects)))

        # DEBUG: Print positions, orientations, and areas of all objects detected in image
        # for j in range(len(objects)):
        #     print('{} : {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(j, objects['x'][j], objects['y'][j], objects['theta'][j] * 180. / np.pi, math.pi * 6 * objects['a'][j] * 6 * objects['b'][j]))


        # After first iteration only (once trackID variable has been incremented at least once), once tracking arrays have been initialised
        if trackID > 1:

            # Loop over all new particle detections
            for j in range(len(objects)):

                # Initialise flag variable to allow addition of new tracks (from objects that cannot be added to any existing track) to track array
                fl = 0

                # Calculate particle size
                size = math.pi * 6 * objects['a'][j] * 6 * objects['b'][j]

                # Only add detection to track (or create new track) if detection is above some minimum size
                if size >= minSize:

                    # Loop over all particle detections in the last tOngoing timesteps (tracks still in the ongoingTracks array)
                    for track in ongoingTracks:

                        # Find latest position on track
                        trackX = track.readXPoint()
                        trackY = track.readYPoint()

                        # Find latest size of object on track
                        trackSize = math.pi * track.reada() * track.readb()

                        # Add object to track if it is close enough in position and size to latest entry of track
                        if (np.sqrt(((trackX - objects['x'][j]) ** 2) + ((trackY - objects['y'][j]) ** 2)) <= maxSep  # Needs to be close in position
                                and trackSize - maxSizeSep <= size <= trackSize + maxSizeSep  # Needs to be close in size
                                and track.readOngoingTime() < i):  # Ensures tracks that have already been updated this timestep cannot have another new particle added

                            # Add object to track
                            track.addPoint((objects['x'][j], objects['y'][j], objects['theta'][j]), 6 * objects['a'][j], 6 * objects['b'][j], i)
                            # DEBUG: print("Track has been appended")

                            # Object has been added to track, so change flag variable
                            fl = 1

                            # Once track has been appended do not want to add object to any other tracks, so break out of loop
                            break


                    # If particle is new (cannot be added to previous track), create a new track for it
                    if fl == 0:

                        # Initialise track using track class defined in file and add to ongoingTracks array
                        track = Track((objects['x'][j], objects['y'][j], objects['theta'][j]), 6 * objects['a'][j], 6 * objects['b'][j], trackID, i)
                        ongoingTracks.append(track)
                        trackID += 1


            # If no particle has been added to a track for tOngoing timesteps then delete track from ongoingTracks array so that it cannot be added to in the future (object being tracked has left)
            # Define buffer to store ongoing tracks (so as tracks are deleted from the actual ongoingTracks array, the indexes of the tracks in the buffer remain the same, and loops remain over all elements of the array)
            ongoingTracks_buff = ongoingTracks[:]

            # Loop over all ongoing tracks
            for track in ongoingTracks_buff:

                TOI = track.readID()

                # If the current time is greater than the time the last object was added to the track plus the allowed gap in time between additions, then the track is complete
                if i > track.readOngoingTime() + tOngoing:

                    # Need to find track in ongoingTracks array (will not necessarily be at the same index as in the buffer), so loop over all entries
                    for k in range(0, len(ongoingTracks)):

                        # If track has same ID number as the one that needs to be removed delete it from ongoingTracks array
                        if ongoingTracks[k].readID() == TOI:

                            del ongoingTracks[k]

                            # Do not need to look at other entries in ongoingTracks as the correct track has been deleted (inceases speed)
                            break

                    # Add track to completeTracks
                    completeTracks.append(track)


        # Else initialise tracking array
        else:

            # All detected objects are starts of tracks
            for j in range(len(objects)):

                # Initialise tracks using track class defined in file
                track = Track((objects['x'][j], objects['y'][j], objects['theta'][j]), 6 * objects['a'][j], 6 * objects['b'][j], trackID, i)

                # Add tracks to list of ongoing tracks
                ongoingTracks.append(track)

                # Each object gets new ID number
                trackID += 1


        # Can plot graphs every frame of film
        if graph1Frame == 1:

            # DEBUG: Plot input data
            # m, s = np.mean(data), np.std(data)
            # print("Mean: {:.4f}, Std: {:.4f}'.format(m, s))
            # plt.imshow(data, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + s, origin='lower')
            # plt.colorbar()
            # plt.show()

            # DEBUG: Plot spatially varying background
            # plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
            # print(1.5 * bkg.globalrms)
            # plt.colorbar()
            # plt.show()

            # Plot background subtracted image
            fig, ax = plt.subplots()
            m, s = np.mean(data_sub), np.std(data_sub)
            im = plt.imshow(data_sub, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + s, origin='lower')
            # plt.colorbar()

            # Plot ellipses fitted around particles
            for j in range(len(objects)):
                e = Ellipse(xy=(objects['x'][j], objects['y'][j]), width=6 * objects['a'][j], height=6 * objects['b'][j], angle=objects['theta'][j] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                e.set_linewidth('1')
                ax.add_artist(e)

            # Plot ongoing tracks on image
            for track in ongoingTracks:
                plt.plot(track.readAllXPoints(), track.readAllYPoints())

            # Can automatically close plot window after a short time so that time series of tracks can be observed
            # plt.xlim([90, 290])
            # plt.ylim([882, 962])
            # plt.xlim([154, 354])
            # plt.ylim([0, 80])
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()


    # DEBUG: After looking at all frames, add all remaining ongoingTrack elements to the completeTracks list (do not need to empty ongoingTracks list as is no longer used)
    # for track in ongoingTracks:
    #     completeTracks.append(track)


    # DEBUG: Count number of times an object was added to a track after a gap of more than one timestep (should be 0 for tOngoing = 0, and increase as a function of tOngoing)
    count = 0
    for track in completeTracks:
        for i in range(1, len(track.points)):
            if track.readTime()[i] > track.readTime()[i - 1] + 1:
                print('ID: {}, times: {}, {}'.format(track.readID(), track.readTime()[i], track.readTime()[i - 1]))
                count += 1
    print('{} addons'.format(count))



# Else read in track data from file (outputted from previous use of code):
else:

    # Loop over all track files
    for i in range(1, numFiles + 1):

        # Open file to read
        varsTrack = open("%sVars_Track%d.txt" % (fileDataIn, i), "r")

        # Initialise track ID number and start time variables (in case reading vars from file is not possible)
        trackID = 0
        trackStart = 0

        # Find track ID number and start time variables from varsTrack files
        count = 1

        # Check file is in read mode
        if varsTrack.mode == 'r':

            # Read all lines from file
            lines = varsTrack.readlines()

            # Loop over lines and access entries as individual numbers
            for lin in lines:

                spl = lin.split()

                # First entry is track ID number
                if count == 1:
                    trackID = int(spl[0])
                # Second entry is start time
                if count == 2:
                    trackStart = int(spl[0])

                count += 1

        # Close file after reading
        varsTrack.close()


        # Repeat to find track positions and angles at each timestep
        dataTrack = open("%sData_Track%d.txt" % (fileDataIn, i), "r")

        count = 1

        if dataTrack.mode == 'r':

            lines = dataTrack.readlines()

            for lin in lines:

                spl = lin.split()

                # First entry requires initialisation of track using track class definition from file
                if count == 1:
                    track = Track((float(spl[0]), float(spl[1]), float(spl[2])), float(spl[3]), float(spl[4]), trackID, trackStart)
                    # Tracks are all already completed, so add straight to completeTracks array
                    completeTracks.append(track)
                # All other entries can append existing track
                else:
                    track.addPoint((float(spl[0]), float(spl[1]), float(spl[2])), float(spl[3]), float(spl[4]), int(spl[5]))

                count += 1

        dataTrack.close()


    # DEBUG: Read in number of 1 frame detections for analysis
    # dataSingleFrame = open("%sN1Frame.txt" % fileTrackDataOut, "r")
    # line = dataSingleFrame.readlines()
    # spl = line[0].split()
    # N1Frame = int(spl[0])



# Plot single particle track of interest with image background:
if graph1ImageTrack == 1 and dataIn == 1:

    # Loop over all tracks
    for track in completeTracks:

        # Only want to plot track of interest
        if track.readID() == TOI:  # This function of the code only possible when reading in data, TOI variable does not interfere with previous use

            # initialise array to store positions of all points on track
            imageTrack = list()

            # Loop over all times where object in track is present
            count = 0
            for i in range(track.readStart(), track.readStart() + track.trackLength()):

                # Read in fits image
                data = fitsio.read("%sImage%d.fits" % (fileImagesIn, i))
                data = data.astype(float)

                # Measure spatially varying background
                bkg = sep.Background(data)
                bkg_image = bkg.back()
                bkg_rms = bkg.rms()

                # Subtract background from data
                data_sub = data - bkg

                # Extract objects from data (after background subtraction)
                objects = sep.extract(data_sub, objErr, err=bkg.globalrms, minarea=minArea)
                print('Image {}, Num. objects: {}'.format(i, len(objects)))

                # Plot background subtracted image
                fig, ax = plt.subplots()
                m, s = np.mean(data_sub), np.std(data_sub)
                im = ax.imshow(data_sub, interpolation='nearest', cmap='gray', vmin=m - s, vmax=m + s, origin='lower')

                # Plot ellipse fitted around particle
                e = Ellipse(xy=(track.readAllXPoints()[count], track.readAllYPoints()[count]), width=track.readAlla()[count] * 2, height=track.readAllb()[count] * 2, angle=track.readAllThetaPoints()[count] * 180. / np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('lime')
                e.set_linewidth(3)

                ax.add_artist(e)

                # Plot track
                imageTrack.append((track.readAllXPoints()[count], track.readAllYPoints()[count]))
                plt.plot(*zip(*imageTrack), 'r-', linewidth=3, alpha=0.7)

                # Save images of individual track (to be compiled into a video using ImageJ software)
                fileTrackImagesOut_buff = "%sImage%d.tif" % (fileTrackImagesOut, count + 1)
                fig.savefig(fileTrackImagesOut_buff, bbox_inches='tight')

                # plt.show(block=False)
                # plt.pause(1.0)
                plt.close()

                count += 1

            # Have found track of interest, do not need to consider the rest of the saved tracks
            break



# Begin analysis #######################################################################################################

# Define a graining for distribution plots to decrease bin width size and increase plot accuracy
graining = 10



# Count number of objects observed as a function of time
if graphNumBound == 1:
    countNumBound(dataOutNumBound, fileDataOut, numImages, frameInit, dt, completeTracks)



# Count number of objects of each area size to find characteristic shape sizes in images
if graphAllAreas == 1:
    peakArea = areasPlot(dataOutAreas, fileDataOut, graphLog, dx, graining, completeTracks)



# Count the number of tracks and add those that fit the requirements to an array for data analysis / outputting
N = 0
bestTrackCount = 0

# Count the number of single frame detections
N1Frame = 0

# Count number of 'unbinding' events that occur within one allowed jump of the edge of the field of view
NEdge = 0
N1Edge = 0  # Stores the number of single frame detections in the edge region of the field of view

# Find width and height of image
data = fitsio.read("%sImage1.fits" % fileImagesIn)
data = data.astype(float)
width = len(data[0])
height = len(data)
# DEBUG: Print width and height of video field of view
# print('Width = {}, Height = {}'.format(len(data[0]), len(data)))

# Calculate mean path length, maximum path length and maximum (straight-line) displacement
meanLen = 0
maxLen = 0
maxLenID = 0
maxDisp = 0
maxDispID = 0

# Loop over all detected tracks
for track in completeTracks:

    # Increment total number of tracks and mean track length
    N += 1
    meanLen += track.trackLength()

    # Calculate average and maximum area size of detected object during track
    avArea = 0
    maxArea = 0
    minArea = 999
    a = np.array(track.readAlla()) * dx  # Convert pixel units to um
    b = np.array(track.readAllb()) * dx
    for i in range(0, track.trackLength()):
        ar = math.pi * a[i] * b[i]
        avArea += ar
        if ar > maxArea:
            maxArea = ar
        if ar < minArea:
            minArea = ar
    avArea /= track.trackLength()

    # DEBUG: Ensure no detections have a size less than the minimum cutoff
    if minArea < minSize * dx * dx:
        print('Detections smaller than minSize threshold included in analysis')


    # Count the number of single frame detections
    if track.trackLength() == 1:  # and (maxArea < 10 * maxSizeSep * dx * dx):
        N1Frame += 1

        # DEBUG: Ensure no single frame detections have been read in from data files
        if dataIn == 1:
            print('Single frame detections read in from data files')

        # Count the number of these that 'unbind' within one jump from the edge of the field of view
        if (track.readXPoint() <= maxSep) or (track.readXPoint() >= width - maxSep) or (track.readYPoint() <= maxSep) or (track.readYPoint() >= height - maxSep):
            N1Edge += 1


    # If track fits specified parameters, store in array for data analysis / outputting
    if (maxLenCutoff >= track.trackLength() > minLenCutoff) and (maxDispCutoff >= track.dispSL() >= minDispCutoff) and (maxArea < 4 * maxSizeSep * dx * dx):

        # Array contains only interesting tracks
        bestTracks.append(track)
        bestTrackCount += 1

        # DEBUG: Ensure no single frame detections are included in main section of analysis
        if track.trackLength() == 1:
            print('Single frame detections included in analysis')

        # Count the number of detections that 'unbind' within one jump from the edge of the field of view
        if (track.readXPoint() <= maxSep) or (track.readXPoint() >= width - maxSep) or (track.readYPoint() <= maxSep) or (track.readYPoint() >= height - maxSep):
            NEdge += 1

        # Find longest track (of those being analysed / outputted for individual videos / plotting of best tracks)
        if track.trackLength() > maxLen:
            maxLen = track.trackLength()
            maxLenID = track.ID
        # Find track with largest (straight-line) displacement
        if track.dispSL() > maxDisp:
            maxDisp = track.dispSL()
            maxDispID = track.ID

# Identify tracks with maximum length and displacement (best for individual videos / plotting)
print('MaxLen track ID:  {}\nMaxDisp track ID: {}'.format(maxLenID, maxDispID))

# Print number of 1 frame detections for diagnostics
print('Number of 1 frame events = {}'.format(N1Frame))

# Print the fraction of these 1 frame events that are also edge events for diagnostics
frac1Edge = 0
if N1Frame > 0:
    frac1Edge = N1Edge / N1Frame
print('Fraction of 1 frame, edge events = {} / {} = {:.4f}'.format(N1Edge, N1Frame, frac1Edge))

# Print fraction of multi-frame edge events for diagnostics
fracEdge = 0
if bestTrackCount > 0:
    fracEdge = NEdge / bestTrackCount
print('Fraction of multi-frame edge events = {} / {} = {:.4f}'.format(NEdge, bestTrackCount, fracEdge))

# Print total number of edge events for diagnostics
fracAllEdge = 0
if bestTrackCount > 0 or N1Frame > 0:
    fracAllEdge = (NEdge + N1Edge) / (bestTrackCount + N1Frame)
print('Fraction of all edge events = {} / {} = {:.4f}'.format(NEdge + N1Edge, bestTrackCount + N1Frame, fracAllEdge))

# Calculate average track length
meanLen = meanLen / N



# Recount number of objects observed as a function of time after removal of some tracks
if graphNumBound == 1:
    countNumBound(dataOutNumBound, fileDataOut, numImages, frameInit, dt, bestTracks)



# Recount number of objects of each area size to find characteristic shape sizes in images after removal of some tracks
if graphAllAreas == 1:
    areasPlot(dataOutAreas, fileDataOut, graphLog, dx, graining, bestTracks)



# Initialise variables for analysis / plotting:

# Displacement / velocity variables:
disp = np.zeros((100, 2))                    # Stores all displacements (instantaneous valocities) for all objects (100 chosen as >> maximum displacement expected between frames, is equal to 10 px jump between frames which is prevented by maxSep)
for i in range(0, 100):
    disp[i, 0] = (dx / graining) * (i + 0.5)
dispComp = list()                            # Stores the components of the displacement parallel and perpendicular to the orientation of the object
MSD = np.zeros((maxLen - 1, 3))              # Stores MSDs for all objects (maxLen is length of track, have (maxLen - 1) displacements)
MSDProc = np.zeros((maxLen - 1, 3))          # Stores MSDs for processive regions of track
MSDDiff = np.zeros((maxLen - 1, 3))          # Stores MSDs for diffusive regions of track
numProc = np.zeros(bestTrackCount)           # Stores number of processive regions on each track
dispTot = list()                             # Stores total straight-line displacements for all tracks
bTot = list()                                # Stores whether track is diffusive or processive in binary at each timestep for all tracks, including those that have no processive regions
dispProc = list()                            # Stores the displacements of each processive region (one data point per processive region) for all objects

# Angular variables:
radii = np.zeros(90)                         # Stores relative probabilities of (orientation angle - processive direction) at each timestep on (pi / 2) domain
radiiDelAng = np.zeros(180)                  # Stores relative probabilities of (orientation angle - processive direction) at each timestep on pi domain
radiiAng = np.zeros((maxLen, 360))           # Stores relative probabilities of orientation angle at each timestep on (2 * pi) degree domain
radiiAng_0 = np.zeros((maxLen, 360))         # Stores relative probabilities of orientation angle at each timestep on (2 * pi) degree domain after initialisation so theta(0) = 0
radiiDelAngProc = np.zeros(180)              # Stores relative probabilities of (orientation angle - processive direction) at each timestep during processive regions on pi domain
radiiDelAngDiff = np.zeros(180)              # Stores relative probabilities of (orientation angle - processive direction) at each timestep during diffusive regions on pi domain

# Initialise plot for graphing of all tracks
if graphAllFinal == 1:
    fig, ax = plt.subplots()



# Gather desired data from all designated tracks:
i = 0          # Labels loops
j = 1          # Labels output files
procCount = 0  # Counts number of processive tracks

# Loop over all tracks to be analysed / outputted:
for track in bestTracks:

    # Define track length
    length = track.trackLength()

    # DEBUG: Print details of track
    # print('Track number {}, length {}, final position ({:.4f}, {:.4f}, {:.4f})'.format(track.ID, length, track.readXPoint(), track.readYPoint(), track.readTheta()))

    # Plot scatter graph of all tracks
    if graphAllFinal == 1:
        ax.plot(np.array(track.readAllXPoints()) * dx, np.array(track.readAllYPoints()) * dx)


    # Find where tracks are diffusive or processive in nature:
    # Approximate whether track has processive region by finding large displacements (over 10 timesteps) that cannot be due to Brownian motion
    proc = track.processiveDefine(dt, dx, diffusivity)

    # Perform analysis on processive region of track only if track has possibility of having a processive region
    if proc == 1:

        # Find processive region(s) of track and store variables in buffer
        proc_buff = track.processiveFind(dt, dx, diffusivity, minLen)

        # Store diagnostic variables for plotting
        qdt1 = proc_buff[0]
        qdt2 = proc_buff[1]
        qs   = proc_buff[2]

        # Store binary array in list
        bTot.append(proc_buff[3])

        # Increment counter for number of tracks with processive regions
        if sum(bTot[i]) != 0:
            procCount += 1

    # Else store null (0) values in bTot if no processive regions were detected on the track
    else:

        bTot.append(np.zeros(length))


    # Calculate the orientation angle, and the difference between the orientation angle and the direction of propagation for each timestep
    radii_buff = track.angles(fileDataOut, maxLen, bTot[i])

    # Store change in orientation and propagation angles on (-pi < angle < pi) domain
    delAng = radii_buff[0]

    # Add histogram array describing number of occurrences of each change in angle to existing array
    radiiDelAng += radii_buff[1]

    # Store orientation angles
    ang = radii_buff[2]

    # Add histogram array describing number of occurrences of each orientation angle to existing array
    radiiAng += radii_buff[3]

    # Store orientation angles initialised to angle[0] = 0
    ang_0 = radii_buff[4]

    # Add histogram array describing number of occurrences of each orientation angle (initialised to angle[0] = 0) to existing array
    radiiAng_0 += radii_buff[5]

    # Add histogram array describing number of occurences of each change in angle during a processive region to existing array
    radiiDelAngProc += radii_buff[6]

    # Add histogram array describing number of occurences of each change in angle during a diffusive region to existing array
    radiiDelAngDiff += radii_buff[7]


    # Calculate object displacement each timestep
    disp_buff = track.displacements(dt, dx, graining, delAng, bTot[i])

    # Store displacements to be plotted in a histogram, loop over possible indexes for plot
    disp[:, 1] += disp_buff[0][:]

    # Store components of displacement parallel and perpendicular to orientation of object
    dispComp.append(disp_buff[1])

    # Store number of processive regions on track
    numProc[i] = disp_buff[3]

    # DEBUG: Identify tracks with multiple (>1) processive regions  for individual videos / plotting
    # if numProc[i] > 0:
    #     print('{} processive region(s), track length: {}, track ID: {}'.format(int(numProc[i]), track.trackLength(), track.ID))


    # Store total displacement during track
    dispTot.append(track.dispSL() * dx)


    # Store total arc length against time for processive regions of track and plot processive regions
    if proc == 1:

        # Store total arc length for processive region(s) of track
        for k in range(0, len(disp_buff[2])):
            dispProc.append(disp_buff[2][k])

        # DEBUG: Plot all processive tracks in turn
        # TOI = track.ID

        # Plot processive and diffusive regions of track with vector arrows indicating orientation
        if graph1Proc == 1 and track.ID == TOI:
            track.processivePlot(dataOutProc, fileDataOut, dt, dx, qdt1, qdt2, qs, bTot[i], ang, delAng)


    # Calculate MSD and MSAD of object (use orientation angle to find MSAD, not delAng)
    MSD_buff = track.MSD(bTot[i], ang)

    # Store average MSD for track
    for k in range(0, len(MSD_buff[0])):
        MSD[k, :] += MSD_buff[0][k, :]

    # DEBUG: Check indexing of MSDProc
    # if len(MSD_buff[1]) > 0:
    #     print('New processive MSD array: {} processive region(s)'.format(len(MSD_buff[1])))
    #     if numProc[i] > 1:
    #         print(MSD_buff[1][0])
    #         print(MSD_buff[1][1])

    # Store separate MSD's for processive and diffusive region(s) of track:
    # Loop over number of processive regions in track
    for k in range(0, len(MSD_buff[1])):

        # DEBUG: Check indexing of MSDProc
        # print('{}'.format(len(MSD_buff[1][k])))

        # Loop over number of MSD entries in the processive region
        for l in range(0, len(MSD_buff[1][k])):

            # Store MSD's for processive region in array indexed by time (number of frames between points displacement was calculated over)
            MSDProc[l, :] += MSD_buff[1][k][l, :]

    # Repeat for diffusive region(s) of track
    for k in range(0, len(MSD_buff[2])):
        for l in range(0, len(MSD_buff[2][k])):
            MSDDiff[l, :] += MSD_buff[2][k][l, :]


    # Output track data as text files
    if dataOut == 1:

        # Output track variables (track ID number and start time)
        f = open("%sVars_Track%d.txt" % (fileTrackDataOut, j), "w")
        f.write('{}\n{}\n'.format(track.readID(), track.readStart()))
        f.close()

        # Output track positions at each timestep
        f = open("%sData_Track%d.txt" % (fileTrackDataOut, j), "w")
        count = 0
        for point in track.points:
            f.write('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {}\n'.format(point[0], point[1], point[2], track.readAlla()[count], track.readAllb()[count], track.readTime()[count]))
            count += 1
        f.close()

        # Increment counter for labelling output files
        j += 1

    # Increment counter for labelling loop
    i += 1


# Output number of single frame tracks as text file
if dataOut == 1:
    f = open("%sN1Frame.txt" % fileTrackDataOut, "w")
    f.write('{}'.format(N1Frame))
    f.close()


# Show plot of all tracks
if graphAllFinal == 1:
    plt.xlabel('$x$ ($\\mu m$)')
    plt.ylabel('$y$ ($\\mu m$)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Take mean of squared displacement to calculate MSD's
#DEBUG: Check MSD array
# print(MSD)
MSD[:, 1] /= MSD[:, 0]
MSD[:, 2] /= MSD[:, 0]
#DEBUG: Check MSD array
# print(MSD)

# Need to ensure there is no division by zero for processive and diffusive arrays (as there will not be enough data to fill all rows)
for i in range(0, maxLen - 1):
    if MSDProc[i, 0] > 0:
        MSDProc[i, 1] /= MSDProc[i, 0]
        MSDProc[i, 2] /= MSDProc[i, 0]
    if MSDDiff[i, 0] > 0:
        MSDDiff[i, 1] /= MSDDiff[i, 0]
        MSDDiff[i, 2] /= MSDDiff[i, 0]


# Take mean and standard deviation of number of processive region(s) on track(s)
meanNumProc = np.mean(numProc)
stdNumProc  = np.std(numProc)

# Take mean and standard deviation of arc lengths for processive regions on tracks
if sum(numProc) > 0:

    # DEBUG: Check indexing of dispProc array
    # print(dispProc)
    # print(len(dispProc))
    # print(dispProc[1])
    # print(len(dispProc[1]))
    # print(dispProc[:][1])
    # print(len(dispProc[:][1]))
    # print(dispProc[1][0])
    # print(len(dispProc[1][0]))
    # print(dispProc[:][1][0])
    # print(len(dispProc[:][1][0]))

    # Find maximum number of processive displacements for a single processive region of track
    maxProcLen = 0

    # Loop over all processive regions of track
    for i in range(0, len(dispProc)):

        # Check if length of processive region is the longest observed so far
        if len(dispProc[i]) > maxProcLen:
            maxProcLen = len(dispProc[i])

    # Print length of longest processive region on a track
    print('Longest processive region (out of {}): {} timesteps'.format(len(dispProc), maxProcLen))


    # Redefine dispProc array for simplified plotting
    dispProc_buff = np.zeros((maxProcLen, 3))

    # Store times for displacements
    dispProc_buff[:, 0] = np.linspace(dt, maxProcLen * dt, maxProcLen)

    # Loop over all processive regions of track
    for i in range(0, len(dispProc)):

        # Loop over all displacements for current processive region of track
        for j in range(0, len(dispProc[i])):

            # DEBUG: Check indexing of dispProc array
            # print('Time: {}'.format(dispProc_buff[j, 0]))
            # print(dispProc[i][j])
            # print(dispProc[i][j, 0])
            # print(dispProc[i][j, 1])

            # Store cumulative sum of contributions for each index of the displacement array, and the number of contributions for each
            dispProc_buff[j, 1] += dispProc[i][j, 1]

            # Record the processive regions number of contributions to this index
            dispProc_buff[j, 2] += dispProc[i][j, 0]

    # Store data in original array
    dispProc = np.zeros((maxProcLen, 2))
    dispProc[:, 0] = dispProc_buff[:, 0]
    dispProc[:, 1] = dispProc_buff[:, 1] / dispProc_buff[:, 2]  # Take average displacement

    # DEBUG: Print resulting distributions
    # print(dispProc_buff)
    # print(dispProc)


# Print final outputs of tracking
print('Number of particles tracked {} with mean track length {:.4f}um (analysing data for {})'.format(N, meanLen * dx, bestTrackCount))
print('{} tracks with (at least one) processive region (average number of processive regions per track ({:.4f} +/- {:.4f})\n'.format(procCount, meanNumProc, stdNumProc))



# Begin plotting #######################################################################################################

# Plot and fit MSD and MSDF curves
if graphAllMSD == 1:

    # Plot and fit MSD and MSAD for all objects (only save MSD and MSAD data for all objects)
    print("MSD from all data:")
    MSDPlot(dataOutMSD, fileDataOut, dt, dx, MSD)

    # Plot and fit MSD and MSAD for processive objects
    if sum(MSDProc[:, 0] > 0):
        print("MSD from processive data:")
        MSDPlot(0, fileDataOut, dt, dx, MSDProc)

    # Plot and fit MSD and MSAD for diffusive objects
    if sum(MSDDiff[:, 0] > 0):
        print("MSD from diffusive data:")
        MSDPlot(0, fileDataOut, dt, dx, MSDDiff)



# Plot displacement distribution for all particles and all times
if graphAllDisp == 1:
    dispPlot(dataOutDisp, fileDataOut, graphLog, disp, dispTot)



# Plot orientation angles as a radial histogram
if graphAllOrientations == 1:

    # Plot all orientation angles as observed (only save orientation angle data for correct initial angle)
    radPlot360(dataOutOrientations, fileDataOut, 0, dt, maxLen, radiiAng)

    # Plot orientation angles after setting initial angle to zero
    radPlot360(0, fileDataOut, graphAllOrientationsAllt, dt, maxLen, radiiAng_0)



# Plot difference between orientation angle and propagation direction as radial histogram
if graphAlldelAng == 1:

    # Plot magnitude of difference in angle on (0 < angle < pi) domain
    radPlot180(dataOutDelAng, fileDataOut, radiiDelAng)

    # DEBUG: Plot minimum difference in angle on (0 < angle < (pi / 2)) domain assuming axial, elliptical symmetry (orientation is symmetric under rotation of pi radians)
    radiiBuff = np.zeros(90)
    for i in range(0, 90):
        radiiBuff[i] = radiiDelAng[i] + radiiDelAng[179 - i]
    radPlot90(radiiBuff)

    # Plot magnitude of difference in angle during all processive regions of track on (0 < angle < pi) domain
    radPlot180(0, fileDataOut, radiiDelAngProc)
    # DEBUG: Print distribution for comparison
    # print('Processive:')
    # print(radiiDelAngProc)

    # Plot magnitude of difference in angle during all diffusive regions of track on (0 < angle < pi) domain
    radPlot180(0, fileDataOut, radiiDelAngDiff)
    # DEBUG: Print distribution for comparison
    # print('Diffusive:')
    # print(radiiDelAngDiff)



# Plot number of occurrences of dwell time and displacement events
if graphAllFreq == 1:
    freqPlot(dataOutTret, dataOutXret, fileDataOut, graphLog, dt, dx, graining, maxLen, maxDisp, N1Frame, procCount, bTot, bestTracks)



# Plot timescales and displacement scales for processive and diffusive regions of tracks
if graphAllProc == 1:

    # Plot variables related to total dwell times and displacements for each state for tracks with a processive region
    procVarPlot(dataOutProc, fileDataOut, dt, bestTrackCount, bTot, numProc, dispProc)

    # Plot displacement variables for processive and diffusive regions of track separately
    procPlot(dataOutProc, fileDataOut, graphAllProc, dx, graining, procCount, bTot, dispComp, disp, bestTracks)



# Plot components of the displacement parallel and perpendicular to the orientation of the object for all objects
if graphAllComp == 1:
    dispComponentsPlot(dataOutComp, fileDataOut, graphLog, dx, graining, dispComp, disp, bestTracks)
