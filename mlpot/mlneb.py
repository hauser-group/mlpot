import numpy as np
from ase.neb import NEB
from ase.optimize.fire import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
from copy import copy


def distance(a1, a2, permute=False):
    """Custom implementation of the distance function without the checks for
    possible rotations, translations or permutations. Argument permute is just
    for compability with standard ase distance function."""
    return np.linalg.norm(a1.get_positions() - a2.get_positions())


def run_neb_on_ml_pes(ml_neb, training_images, optimizer=FIRE, fmax=0.5,
                      steps=250, r_max=1.0):
    """
    Runs the NEB optimization on the machine learning PES. Needs an instance of
    NEB with the machine learning calculators attached to them. The
    training_images are needed to stop the optimization as soon as any image
    moves more than r_max distance units.
    Returns boolean of convergence status."""
    print('Starting machine learning neb run.')
    opt = optimizer(ml_neb)
    old_positions = [image.get_positions() for image in ml_neb.images[1:-1]]
    # loop over optimizer steps
    for converged in opt.irun(fmax=fmax, steps=steps):
        # check if any image has moved to far from the training_images
        for ni, ml_image in enumerate(ml_neb.images[1:-1]):
            distances = np.zeros(len(training_images))
            for nj, training_image in enumerate(training_images):
                distances[nj] = distance(ml_image, training_image)
            # stop if the distance from ml_image to the closest training_image
            # is larger than r_max
            if np.min(distances) > r_max:
                print('Image %d exceeded r_max at step %d.' % (
                    ni, opt.nsteps), 'Resetting to previous step.')
                [ml_image.set_positions(old_pos.copy())
                 for ml_image, old_pos
                 in zip(ml_neb.images[1:-1], old_positions)]
                return False, ni
        old_positions = [image.get_positions()
                         for image
                         in ml_neb.images[1:-1]]
    return converged, None


def run_mla_neb(neb, ml_calc, optimizer=FIRE, steps=100, f_max=0.05,
                f_max_ml=None, f_max_ml_ci=None, steps_ml=250, steps_ml_ci=150,
                r_max=None, callback_after_ml_neb=None):
    """
    """
    images = neb.images
    N_atoms = len(images[0])
    # save initial path as the machine learning NEB run is always restarted
    # from the initial path.
    initial_path = [image.get_positions().copy() for image in images]

    # set default values for the machine learning NEB calculations
    f_max_ml = f_max_ml or 10*f_max
    f_max_ml_ci = f_max_ml_ci or 0.5*f_max
    if r_max is None:
        # Koistinen et al. J. Chem. Phys. 147, 152720 (2017) suggest half of
        # the length of the initial path for r_max:
        r_max = 0.5*sum(
            [distance(images[i-1], images[i]) for i in range(1, len(images))])
        print('r_max = %.2f' % r_max)

    # make a copy of all images and attach a copy of the machine learning
    # calculator. Add a copy the whole band to the training images
    ml_images = [image.copy() for image in images]
    training_images = [image.copy() for image in images]
    for image, ml_image, training_image in zip(images, ml_images,
                                               training_images):
        ml_image.set_calculator(copy(ml_calc))
        training_image.set_calculator(SinglePointCalculator(
            energy=image.get_potential_energy(),
            forces=image.get_forces(apply_constraint=False),
            atoms=training_image))
        ml_calc.add_data(training_image)

    ml_neb = NEB(
        ml_images,
        remove_rotation_and_translation=neb.remove_rotation_and_translation)

    for step_i in range(steps):
        # get the forces on the inner images including the spring forces and
        # reshape them
        true_forces = neb.get_forces().reshape((len(neb.images)-2, N_atoms, 3))
        print('Maximum force per image after %d evaluations of the band:' % (
            step_i+1))
        print(np.sqrt((true_forces**2).sum(axis=2).max(axis=1)))
        # Check for convergence, following the default ase convergence
        # criterion
        if (true_forces**2).sum(axis=2).max() < f_max**2:
            print('Converged after %d evaluations of the band.' % (step_i+1))
            break

        # fit the machine learning model to the training images
        ml_calc.fit()
        # save the fitted parameters of the machine learning model
        params = ml_calc.get_params()

        # reset machine learning path to initial path and set the parameters of
        # the individual ml_image calculators to the newly fitted values.
        for ml_image, init_positions, image in zip(ml_images, initial_path,
                                                   images):
            ml_image.set_positions(init_positions.copy())
            ml_image.calc.set_params(**params)

        # optimize nudged elastic band on the machine learning PES. Start
        # without climbing. Should the first run converge. Switch climb = True
        # and optimize.
        ml_neb.climb = False
        if run_neb_on_ml_pes(ml_neb, training_images, optimizer=optimizer,
                             fmax=f_max_ml, steps=steps_ml, r_max=r_max)[0]:
            print('Switching to climbing image NEB')
            ml_neb.climb = True
            run_neb_on_ml_pes(ml_neb, training_images, optimizer=optimizer,
                              fmax=f_max_ml_ci, steps=steps_ml_ci, r_max=r_max)

        if callback_after_ml_neb is not None:
            callback_after_ml_neb(images, ml_images, ml_calc)

        # calculate the inner images at machine learning minimum energy path
        # and append the results to the training images
        for image, ml_image in zip(images[1:-1], ml_images[1:-1]):
            image.set_positions(ml_image.get_positions())
            training_image = image.copy()
            # calculation of the ab initio forces happens at this point because
            # get_potential_energy() and get_forces() are called for the new
            # positions of 'image'.
            training_image.set_calculator(SinglePointCalculator(
                energy=image.get_potential_energy(),
                forces=image.get_forces(apply_constraint=False),
                atoms=training_image))
            training_images.append(training_image)
            ml_calc.add_data(training_image)


def aie_ml_neb(neb, ml_calc, t_MEP, t_CI, t_CIon, r_max):

    def evaluate_image(index):
        

    for i_step in range(steps):
        # Step B:
        true_forces = neb.get_forces()

        # Step C:
        if abs

        # Step D:
        ml_calc.fit()
        params = ml_calc.get_params()


def oie_ml_neb(neb, ml_calc, optimizer=FIRE, steps=100, f_max=0.05,
               f_max_ml=None, f_max_ml_ci=None, steps_ml=250, steps_ml_ci=150,
               r_max=None, callback_after_ml_neb=None):
    images = neb.images
    N_atoms = len(images[0])
    images_evaluated = np.full(len(images), False)
    # save initial path as the machine learning NEB run is always restarted
    # from the initial path.
    initial_path = [image.get_positions().copy() for image in images]

    # make a copy of all images and attach a copy of the machine learning
    # calculator. Add only first and last image to the training data.
    ml_images = []
    training_images = []
    for i, image in enumerate(images):
        ml_image = image.copy()
        ml_image.set_calculator(copy(ml_calc))
        ml_images.append(ml_image)

        if (not image.calc.calculation_required(image, ['energy', 'forces']) or
                i == 0 or i == len(images)):
            training_image = image.copy()
            training_image.set_calculator(
                SinglePointCalculator(
                    energy=image.get_potential_energy(),
                    forces=image.get_forces(apply_constraint=False),
                    atoms=training_image))
            images_evaluated[i] = True
            training_images.append(training_image)
            ml_calc.add_data(training_image)

    ml_neb = NEB(
        ml_images,
        remove_rotation_and_translation=neb.remove_rotation_and_translation)

    # Step 1: fit the machine learning model to the training images
    ml_calc.fit()
    # save the fitted parameters of the machine learning model
    params = ml_calc.get_params()

    # Step A: determine unevaluated image with highest uncertainty
    var_max = 0.
    var_max_i = None
    for i, ml_image in enumerate(ml_images):
        if not images_evaluated[i]:
            ml_image.calc.set_params(**params)
            var = ml_image.predict_var()
            if var > var_max:
                var_max = var
                var_max_i = i

    # Step B: evaluate image with highest uncertainty and add to training data
    if var_max_i is not None:
        training_image = image[var_max_i].copy()
        training_image.set_calculator(
            SinglePointCalculator(
                energy=image[var_max_i].get_potential_energy(),
                forces=image[var_max_i].get_forces(apply_constraint=False),
                atoms=training_image))
        ml_calc.add_data(training_image)

    for step_i in range(steps):
        # Step C:

        # Step D: check for convergence:
        if np.all(images_evaluated):
            if (neb.get_forces**2).sum(axis=1).max() < f_max**2:
                return

        # Step E: refit the machine learning model:
        ml_calc.fit()
        # save the fitted parameters of the machine learning model
        params = ml_calc.get_params()

        # Step F:
        tmp_neb = NEB(
            [im if eval_i else ml_im for eval_i, im, ml_im in zip(
                images_evaluated, images, ml_images)])
        approx_forces = tmp_neb.get_forces()

        # Step G:
        if (approx_forces**2).sum(axis=2).max() < f_max**2:
            if not images_evaluated[tmp_neb.imax]:  # Substep I
                images[tmp_neb.imax].set_positions(
                    ml_images[tmp_neb.imax].get_positions())
                training_image = images[tmp_neb.imax].copy()
                training_image.set_calculator(SinglePointCalculator(
                    energy=image.get_potential_energy(),
                    forces=image.get_forces(apply_constraint=False),
                    atoms=training_image))
                training_images.append(training_image)
                ml_calc.add_data(training_image)
                continue  # Go to C
            elif images_evaluated[tmp_neb.imax] and :  # Substep II
                pass
        # Substep III
        # Step H: Relaxation phase
        # reset machine learning path to initial path and set the parameters of
        # the individual ml_image calculators to the newly fitted values.
        for ml_image, init_positions, image in zip(ml_images, initial_path,
                                                   images):
            ml_image.set_positions(init_positions.copy())
            ml_image.calc.set_params(**params)

        # optimize nudged elastic band on the machine learning PES. Start
        # without climbing. Should the first run converge, switch climb = True
        # and optimize.
        ml_neb.climb = False
        converged, ind = run_neb_on_ml_pes(ml_neb, training_images,
                                           optimizer=optimizer, fmax=f_max_ml,
                                           steps=steps_ml, r_max=r_max)
        if converged:  # Start climing image run
            print('Switching to climbing image NEB')
            ml_neb.climb = True
            converged, ind = run_neb_on_ml_pes(
                ml_neb, training_images, optimizer=optimizer, fmax=f_max_ml_ci,
                steps=steps_ml_ci, r_max=r_max)

        if callback_after_ml_neb is not None:
            callback_after_ml_neb(images, ml_images, ml_calc)

        # Evaluate climbing image if converged, image that caused stopping else
        ind = ml_neb.imax if converged else ind
        images[ind].set_positions(ml_images[ind].get_positions)
        training_image = images[ind].copy()
        training_image.set_calculator(SinglePointCalculator(
            energy=image.get_potential_energy(),
            forces=image.get_forces(apply_constraint=False),
            atoms=training_image))
        training_images.append(training_image)
        ml_calc.add_data(training_image)
