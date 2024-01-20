
try:
    import ants
except:
    pass

# fi = ants.image_read(ants.get_ants_data('r16'))
# mi = ants.image_read(ants.get_ants_data('r64'))
# fi = ants.resample_image(fi, (60,60), 1, 0)
# mi = ants.resample_image(mi, (60,60), 1, 0)
# mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'SyN' )