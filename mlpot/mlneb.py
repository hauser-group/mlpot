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
                return False, ni + 1
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
        for ml_image, init_positions in zip(ml_images, initial_path):
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


def _relaxation_phase(ml_neb, ml_calc, steps, t_mep_ml, t_ci_on, r_max,
                      optimizer=FIRE):
    opt = optimizer(ml_neb)

    old_positions = [image.get_positions() for image in ml_neb.images[1:-1]]
    for converged in opt.irun(fmax=t_mep_ml, steps=steps):
        # Step I and II: get forces and reshape them into
        # N_intermediate_images x N_atoms x 3
        forces = ml_neb.get_forces().reshape((len(ml_neb.images)-2, -1, 3))
        # This differs from Koistinen et al. by checking the maximum force
        # on any atom and not the norm of the force vector
        f_max = np.sqrt((forces**2).sum(axis=2).max())
        # Check for convergence or switch to climbing
        if not ml_neb.climb and f_max < t_ci_on:  # Step III
            print('Turning on climbing mode.')
            ml_neb.climb = True
            return _relaxation_phase(ml_neb, ml_calc, steps - opt.nsteps,
                                     t_mep_ml, t_ci_on, r_max)
        if ml_neb.climb and f_max < t_mep_ml:  # Step IV:
            return True, None

        # Step VI: check if any image has moved to far from the training_images
        for ni, ml_image in enumerate(ml_neb.images[1:-1]):
            distances = np.zeros(len(ml_calc.atoms_train))
            for nj, training_image in enumerate(ml_calc.atoms_train):
                distances[nj] = distance(ml_image, training_image)
            # stop if the distance from ml_image to the closest training_image
            # is larger than r_max
            if np.min(distances) > r_max:
                print('Image %d exceeded r_max at step %d.' % (
                    ni + 1, opt.nsteps), 'Resetting to previous step.')
                [ml_image.set_positions(old_pos.copy())
                    for ml_image, old_pos
                    in zip(ml_neb.images[1:-1], old_positions)]
                return False, ni + 1
        old_positions = [image.get_positions()
                         for image
                         in ml_neb.images[1:-1]]


def aie_ml_neb(neb, ml_calc, steps=150, ml_steps=150, t_mep=0.3, t_ci=0.01,
               t_ci_on=1.0, r_max=None, t_mep_ml=None,
               callback_after_ml_neb=None):
    """


    Koistinen et al. J. Chem. Phys. 147, 152720 (2017)
    """
    images = neb.images
    # save initial path as the machine learning NEB run is always restarted
    # from the initial path.
    initial_path = [image.get_positions().copy() for image in images]

    if r_max is None:
        # Koistinen et al. suggest half of the length of the initial path for
        # r_max:
        r_max = 0.5*sum(
            [distance(images[i-1], images[i]) for i in range(1, len(images))])
        print('r_max = %.2f' % r_max)

    # Default value of the threshold for the MEP on the machine learning
    # surface following Koistinen et al.
    t_mep_ml = t_mep_ml or 0.1*t_ci

    # Add first and last image to the training data
    for image in (images[0], images[-1]):
        training_image = image.copy()
        training_image.set_calculator(SinglePointCalculator(
            training_image,
            energy=image.get_potential_energy(),
            forces=image.get_forces(apply_constraint=False)))
        ml_calc.add_data(training_image)

    ml_images = [image.copy() for image in images]
    [ml_image.set_calculator(copy(ml_calc)) for ml_image in ml_images]
    ml_neb = NEB(
        ml_images,
        remove_rotation_and_translation=neb.remove_rotation_and_translation)

    for i_step in range(steps):
        # Step A:
        # evaluate intermediate images and add them to the training_image data
        for image, ml_image in zip(images[1:-1], ml_images[1:-1]):
            # Update image positions
            image.set_positions(ml_image.get_positions())
            training_image = image.copy()
            training_image.set_calculator(
                SinglePointCalculator(
                    training_image,
                    energy=image.get_potential_energy(),
                    forces=image.get_forces(apply_constraint=False))
            )
            ml_calc.add_data(training_image)

        # Step B:
        # Reshape forces into N_intermediate_images x N_atoms x 3
        forces = neb.get_forces().reshape((len(ml_neb.images)-2, -1, 3))
        print('Maximum force per image after %d evaluations of the band:' % (
            i_step+1))
        print(np.sqrt((forces**2).sum(axis=2).max(axis=1)))

        # Step C:
        # This differs from Koistinen et al. by checking the maximum force
        # on any atom and not the norm of the force vector
        max_force = np.sqrt((forces**2).sum(axis=2).max())
        # Use imax-1 since forces only contains intermediate images
        ci_force = np.sqrt((forces[neb.imax-1, :, :]**2).sum(axis=1).max())
        print('Maximum force: ', max_force)
        print('Force on climbing image: ', ci_force)
        if max_force < t_mep and ci_force < t_ci:
            # Converged
            return True
        # Step D:
        ml_calc.fit()
        params = ml_calc.get_params()

        # Step E:
        for ml_image, init_pos in zip(ml_images, initial_path):
            # Update calculator
            ml_image.calc.set_params(**params)
            # Reset positions to inital path
            ml_image.set_positions(init_pos.copy())
        ml_neb.climb = False
        _relaxation_phase(ml_neb, ml_calc, ml_steps, t_mep_ml, t_ci_on, r_max)

        if callback_after_ml_neb is not None:
            callback_after_ml_neb(images, ml_images, ml_calc)
    # No convergence reached:
    return False


def oie_ml_neb(neb, ml_calc, optimizer=FIRE, steps=100, ml_steps=150,
               t_mep=0.3, t_ci=0.01, t_ci_on=1.0, r_max=None, t_mep_ml=None,
               callback_after_ml_neb=None):
    images = neb.images
    # save initial path as the machine learning NEB run is always restarted
    # from the initial path.
    initial_path = [image.get_positions().copy() for image in images]

    if r_max is None:
        # Koistinen et al. suggest half of the length of the initial path for
        # r_max:
        r_max = 0.5*sum(
            [distance(images[i-1], images[i]) for i in range(1, len(images))])
        print('r_max = %.2f' % r_max)

    # Default value of the threshold for the MEP on the machine learning
    # surface following Koistinen et al.
    t_mep_ml = t_mep_ml or 0.1*t_ci

    def eval_image(ind):
        training_image = images[ind].copy()
        training_image.set_calculator(SinglePointCalculator(
            atoms=training_image,
            energy=images[ind].get_potential_energy(),
            forces=images[ind].get_forces(
                apply_constraint=False)))
        ml_calc.add_data(training_image)

    # Add first and last image as well as any image that does not require
    # recalculation to the training data.
    for i, image in enumerate(images):
        if (not image.calc.calculation_required(image, ['energy', 'forces']) or
                i == 0 or i == len(images)-1):
            eval_image(i)

    # make a copy of all images and attach a copy of the machine learning
    # calculator.
    ml_images = [image.copy() for image in images]
    [ml_image.set_calculator(copy(ml_calc)) for ml_image in ml_images]
    ml_neb = NEB(
        ml_images,
        remove_rotation_and_translation=neb.remove_rotation_and_translation)

    # Step 1: fit the machine learning model to the training images
    print('Step 1')
    ml_calc.fit()
    params = ml_calc.get_params()
    [ml_image.calc.set_params(**params) for ml_image in ml_images]

    def eval_highest_variance():
        # Step A: determine unevaluated image with highest uncertainty
        print('Step A')
        vars = np.zeros(len(images))
        for i, (image, ml_image) in enumerate(zip(images, ml_images)):
            if image.calc.calculation_required(image, ['energy', 'forces']):
                # Calculate variance of the energy prediction:
                vars[i] = ml_image.calc.predict_var(ml_image)[0]
        if np.any(vars < 0.):
            print('Negative variance found. Using absolute values to ' +
                  'determine next image to evalute.')
            vars = np.abs(vars)
        var_max_i = np.argmax(vars)
        # Step B: evaluate image with highest uncertainty and add to training
        # data.
        print('Step B')
        eval_image(var_max_i)

    def step_H():
        # reset machine learning path to initial path
        for ml_image, init_pos in zip(ml_images, initial_path):
            # Reset positions to inital path
            ml_image.set_positions(init_pos.copy())
        ml_neb.climb = False
        converged, ind = _relaxation_phase(ml_neb, ml_calc, ml_steps, t_mep_ml,
                                           t_ci_on, r_max)

        if callback_after_ml_neb is not None:
            callback_after_ml_neb(images, ml_images, ml_calc)

        # Update positions
        [image.set_positions(ml_image.get_positions())
         for image, ml_image in zip(images, ml_images)]
        return converged, ind

    eval_highest_variance()

    for step_i in range(steps):
        # Step C:
        print('Step C')

        # Step D: check for convergence:
        print('Step D')
        if not np.any([im.calc.calculation_required(im, ['energy', 'forces'])
                       for im in images]):
            forces = neb.get_forces().reshape((len(ml_neb.images) - 2, -1, 3))
            # Use imax-1 since forces only contains intermediate images
            if ((forces**2).sum(axis=2).max() < t_mep**2 and
                    (forces[neb.imax-1, :, :]**2).sum(axis=1).max() < t_ci):
                print('Converged. Final number of training points:',
                      len(ml_calc.atoms_train))
                return True

        # Step E: refit the machine learning model:
        print('Step E')
        ml_calc.fit()
        params = ml_calc.get_params()
        [ml_image.calc.set_params(**params) for ml_image in ml_images]

        # Step F:
        print('Step F')
        evaluated_images = [im.calc.calculation_required(
                                im, ['energy', 'forces']) for im in images]
        tmp_neb = NEB(
            [ml_im if eval else im for eval, im, ml_im in zip(
                evaluated_images, images, ml_images)])
        approx_forces = tmp_neb.get_forces().reshape(
            (len(ml_neb.images) - 2, -1, 3))
        print('Maximum force on a atom (in eV/A) for each image, * indicates '
              'approximation by machine learning model')
        print(' '.join(['%.4f*' % f if eval else '%.4f' % f for eval, f in zip(
                evaluated_images[1:-1],
                np.sqrt((approx_forces**2).sum(axis=2).max(axis=1)))]))

        # Step G:
        print('Step G')
        if (approx_forces**2).sum(axis=2).max() < t_mep**2:
            if images[tmp_neb.imax].calc.calculation_required(
                    images[tmp_neb.imax], ['energy', 'forces']):  # Substep I
                print('Step GI')
                eval_image(tmp_neb.imax)
                continue  # Go to C
            elif ((approx_forces[tmp_neb.imax-1, :, :]**2).sum(axis=1).max() <
                  t_ci**2):  # Substep II
                print('Step GII')
                eval_highest_variance()
                continue  # Go to C
            else:  # Substep III
                print('Step GIII')
                converged, ind = step_H()
                # In case of early stopping evaluate image that caused early
                # stopping
                if not converged:
                    eval_image(ind)
                else:  # Evaluate climbing image
                    eval_image(ml_neb.imax)

        # Step H: Relaxation phase
        print('Step H')
        converged, ind = step_H()
        # In case of early stopping evaluate image that caused early stopping
        if not converged:
            eval_image(ind)
        else:  # Evaluate image with highest uncertainty
            eval_highest_variance()
    # No convergence reached
    return False
