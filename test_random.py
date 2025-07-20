
import numpy as np
import matplotlib.pyplot as plt

def Sharpness_DIN45692_from_loudness(SpecificLoudness, weight_type, time=None, time_skip=0, show_sharpness=None):
    # function OUT = Sharpness_DIN45692_from_loudness(SpecificLoudness, weight_type, time, time_skip, show_sharpness)
    #
    #  Stationary and time-varying sharpness calculation according to DIN 45692(2009)
    #  from input specific loudness (i.e. the loudness calculation is not included within this code)
    #
    ###########################################################################
    #
    # INPUT ARGUMENTS
    #   SpecificLoudness : array
    #   if method = 0 (stationary) - Specific loudness [1,sone/Bark]
    #   if method = 1 (time-varying) - Instantaneous specific loudness [nTimeSteps,sone/Bark]
    #
    #   weight_type : string
    #       weighting function used for sharpness calculation, according to:
    #       - 'DIN45692'
    #       - 'bismarck'
    #       - 'aures' (dependent on the specific loudness level)
    #
    #   time : array
    #       time vector of the specific loudness [1,nTimeSteps] - used only for
    #       plot purposes if method = 1 (time-varying)
    #
    #   time_skip : integer
    #   skip start of the signal in <time_skip> seconds for statistics 
    #       calculations (method=1 (time-varying) only)
    #
    #   show : logical(boolean)
    #   optional parameter for figures (results) display (only method=1)
    #   'false' (disable, default value) or 'true' (enable).
    #
    # OUTPUTS (method==0; stationary)
    #   OUT : struct containing the following fields
    #
    #       * Sharpness: sharpness (acum)
    #
    # OUTPUTS (method==1; time-varying)
    #   OUT : struct containing the following fields
    #
    #       * InstantaneousSharpness: instantaneous sharpness (acum) vs time
    #       * time : time vector in seconds
    #       * Several statistics based on the InstantaneousSharpness (acum)
    #         ** Smean : mean value of InstantaneousSharpness (acum)
    #         ** Sstd : standard deviation of InstantaneousSharpness (acum)
    #         ** Smax : maximum of InstantaneousSharpness (acum)
    #         ** Smin : minimum of InstantaneousSharpness (acum)
    #         ** Sx : sharpness value exceeded during x percent of the time (acum)
    #
    #           *** HINT: time-varying loudness calculation takes some time to
    #                     have a steady-response (thus sharpness too!). 
    #                     Therefore, it is a good practice to consider a 
    #                     time_skip to compute the statistics
    #
    # Author: Gil Felix Greco, Braunschweig 09.03.2023
    # Author: Gil Felix Greco, Braunschweig 16.02.2025 - introduced get_statistics function
    ###########################################################################
    
    # Handle default parameters (mimicking Matlab's nargin behavior)
    if show_sharpness is None:
        # In Python, we approximate the nargout==0 check by defaulting to False
        # This can be overridden by explicitly passing True
        show_sharpness = False

    SpecificLoudness = np.array(SpecificLoudness)
    n = SpecificLoudness.shape[1]
    z = np.linspace(0.1, 24, n)   # create bark axis

    if SpecificLoudness.shape[0] == 1: # define method based on the size of the input specific loudness
        method = 0 # (stationary) - Specific loudness [1,sone/Bark]
    else:
        method = 1 # (time-varying) - Instantaneous specific loudness [nTimeSteps,sone/Bark]

    loudness_sones = np.zeros(SpecificLoudness.shape[0]) # pre allocate memory

    for i in range(SpecificLoudness.shape[0]):
        loudness_sones[i] = np.sum(SpecificLoudness[i,:]) * 0.10

    ###########################################################################
    # Sharpness calculation

    s = np.zeros(SpecificLoudness.shape[0])

    if weight_type == 'DIN45692': # Widmann model
        
        g = il_sharpWeights(z,'standard',None) # calculate sharpness weighting factors
        k = 0.11 # adjusted to yield 1 acum using SQAT - DIN45692 allows 0.105<=k<=0.0115 for this weighting function
        
        for i in range(SpecificLoudness.shape[0]):
            s[i] = k * np.sum(SpecificLoudness[i,:]*g*z*0.10) / loudness_sones[i]
        ###########################################################################
        
    elif weight_type == 'aures': # Aures model
        
        g = np.zeros((SpecificLoudness.shape[0], len(z)))
        for i in range(SpecificLoudness.shape[0]):
            g[i,:] = il_sharpWeights(z,'aures',loudness_sones[i]) # calculate sharpness weighting factor
            s[i] = 0.11 * np.sum(SpecificLoudness[i,:]*g[i,:]*z*0.10) / loudness_sones[i]
        
        ###########################################################################
    elif weight_type == 'bismarck': # von Bismarck
        g = il_sharpWeights(z,'bismarck',None) # calculate sharpness weighting factor
        
        for i in range(SpecificLoudness.shape[0]):
            s[i] = 0.11 * np.sum(SpecificLoudness[i,:]*g*z*0.10) / loudness_sones[i]
        
        ###########################################################################

    ###########################################################################
    #   output struct for time-varying signals

    OUT = {}  # Python dictionary to mimic Matlab struct

    if method == 1: # (time-varying sharpness)
        
        OUT['InstantaneousSharpness'] = s # instantaneous sharpness
        OUT['time'] = time                # time vector
        
        # get statistics from Time-varying sharpness (acum)
        #########################################

        idx = np.argmin(np.abs(time - time_skip)) # find idx of time_skip on time vector

        metric_statistics = 'Sharpness_DIN45692'
        OUT_statistics = get_statistics(s[idx:], metric_statistics) # get statistics

        # copy fields of <OUT_statistics> struct into the <OUT> struct
        fields_OUT_statistics = list(OUT_statistics.keys())  # Get all field names in OUT_statistics

        for i in range(len(fields_OUT_statistics)):
            fieldName = fields_OUT_statistics[i]
            if fieldName not in OUT: # Only copy if OUT does NOT already have this field
                OUT[fieldName] = OUT_statistics[fieldName]

        del OUT_statistics, metric_statistics, fields_OUT_statistics, fieldName
        #########################################
        
        #######################################################################
        # Show plots (time-varying)
        #######################################################################
        
        if show_sharpness == True:
            
            plt.figure()
            plt.gcf().canvas.manager.set_window_title('Sharpness analysis (time-varying)')
            
            plt.plot(time, OUT['S5']*np.ones(len(time)), 'r--', hold=True)
            plt.plot(time, s)
            
            plt.xlabel('Time, $t$ (s)')
            plt.ylabel('Sharpness, $S$ (acum)')
            
            plt.legend([f'$S_5$={OUT["S5"]:.3g}'], loc='best')
            
            plt.gca().set_facecolor('white')
            plt.gcf().patch.set_facecolor('white')
            
            plt.show()
        
    elif method == 0: # (stationary sharpness)
        
        OUT['Sharpness'] = s[0]                       # sharpness
        
    return OUT # end of function


def il_sharpWeights(z, type_weight, N):

    g = np.zeros(len(z))

    if type_weight == 'standard': # Widmann model according to DIN 45692 (2009)
        g[z<15.8] = 1
        mask = z>=15.8
        g[mask] = 0.15*np.exp(0.42*(z[mask]-15.8)) + 0.85

    elif type_weight == 'bismarck': # von bismark's model according to DIN 45692 (2009)
        g[z<15] = 1
        mask = z>=15
        g[mask] = 0.2*np.exp(0.308*(z[mask]-15)) + 0.8

    elif type_weight == 'aures':    # Aures' model according to DIN 45692 (2009)
        # Note: N is a single value for this case
        g = 0.078*(np.exp(0.171*z)/z)*(N/np.log(0.05*N+1))

    return g

    """
    Placeholder implementation for get_statistics function.
    This function is called in the original Matlab code but not defined there.
    A complete implementation would need to be provided based on the specific 
    requirements for statistical analysis.
    """
    # This is a minimal implementation to prevent errors
    # The actual implementation should provide statistics like:
    # Smean, Sstd, Smax, Smin, S1, S5, S10, S50, S90, S95, S99
    
    stats = {}
    stats['Smean'] = np.mean(data)
    stats['Sstd'] = np.std(data)
    stats['Smax'] = np.max(data)
    stats['Smin'] = np.min(data)
    
    # Percentile values (these would typically be calculated differently)
    stats['S1'] = np.percentile(data, 99)
    stats['S5'] = np.percentile(data, 95)  
    stats['S10'] = np.percentile(data, 90)
    stats['S50'] = np.percentile(data, 50)
    stats['S90'] = np.percentile(data, 10)
    stats['S95'] = np.percentile(data, 5)
    stats['S99'] = np.percentile(data, 1)
    
    return stats