is_debug = False # Actually verbose prints
debug_show_plots = False  # May require "is_debug = True"

min_pts_for_correlation = 5

two_dim = {
    'r': 0.3,
    'r_step': 0.05,
    'max_collect2_iterations': 50,

    #'r': 100.0,
    #'r_step': 0.15,
    #'max_collect2_iterations': 2,

    'min_correlation': 0.7,
    'remove_low_correlation_pts': True,
}

three_dim = {
    'r': 4.0,
    'r_step': 0.3,
    'max_collect2_iterations': 4,
    'remove_low_correlation_pts': True
}

