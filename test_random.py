
import numpy as np

def PsychoacousticAnnoyance_Di2016_from_percentile(N, S, R, FS, K):
    """
    function OUT = PsychoacousticAnnoyance_Di2016_from_percentile(N,S,R,FS,K)
    
    This function calculates the Di's modified psychoacoustic annoyance model from scalar inputs
    corresponding to the percentile values of loudness, sharpness, roughness, fluctuation strength and tonality
    
    The modified psychoacoustic annoyance model is according to:
    [1] Di et al., Improvement of Zwicker's psychoacoustic annoyance model aiming at tonal noises, Applied Acoustics 105 (2016) 164-170
    
    - This metric combines 5 psychoacoustic metrics to quantitatively describe annoyance:
    
       1) Loudness, N (sone)
    
       2) Sharpness, S (acum)
    
       3) Roughness, R (asper)
    
       4) Fluctuation strength, FS (vacil)
    
       5) Tonality, K (t.u.)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    INPUT:
      N: scalar
      loudness percentile value (sone)
    
      S: scalar
      sharpness percentile value (acum)
    
      R: scalar
      roughness percentile value (asper)
    
      FS: scalar
      fluctuation strength percentile value (vacil)
    
      K: scalar
      tonality percentile value (t.u.)
    
    OUTPUTS:
      OUT : scalar
      modified psychoacoustic annoyance computed using the input percentile values of each metric
    
    Author: Gil Felix Greco, Braunschweig 05.04.2023
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    ## modified PA model constants (Ref. [1] pg. 168, eq (9))
    
    alpha = 0.52
    beta = 6.41
    
    ## (scalar) modified psychoacoustic annoyance - computed directly from percentile values
    
    # sharpness and loudness influence
    if S > 1.75:
        ws = (S - 1.75) * (np.log10(N + 10)) / 4  # in the Fastl&zwicker book, ln is used but it is not clear if it is natural log or log10, but most of subsequent literature uses log10
    else:
        ws = 0
    
    if np.isinf(ws) or np.isnan(ws):  # replace inf and NaN with zeros
        ws = 0
    
    # influence of roughness and fluctuation strength
    wfr = (2.18 / (N ** 0.4)) * (0.4 * FS + 0.6 * R)
    
    if np.isinf(wfr) or np.isnan(wfr):  # replace inf and NaN with zeros
        wfr = 0
    
    # Tonality influence
    wt = (beta / (N ** alpha)) * K
    
    if np.isinf(wt) or np.isnan(wt):  # replace inf and NaN with zeros
        wt = 0
    
    # Di's modified psychoacoustic annoyance
    PA_scalar = N * (1 + np.sqrt(ws**2 + wfr**2 + wt**2))
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   OUTPUT
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # main output results
    
    OUT = PA_scalar  # Annoyance calculated from the percentiles of each variable
    
    return OUT  # end PA function

