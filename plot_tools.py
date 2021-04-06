import matplotlib.pyplot as plt

def set_size(width,hratio=1, fraction=1, subplots=(1, 1)):
	
	width_pt = width
	# Width of figure (in pts)
	fig_width_pt = width_pt * fraction
	# Convert from pt to inches
	inches_per_pt = 1 / 72.27

	golden_ratio = (5**.5 - 1) / 2

	# Figure width in inches
	fig_width_in = fig_width_pt * inches_per_pt
	# Figure height in inches
	fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])*hratio

	return (fig_width_in, fig_height_in)