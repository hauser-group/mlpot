#include "descriptorset.h"
#include <stdio.h>
#include <limits>
#include <math.h>
#include <algorithm>
#include <omp.h>

DescriptorSet::DescriptorSet(int num_atomtypes):
num_descriptors(2*num_atomtypes),
two_body_descriptors(num_atomtypes*num_atomtypes),
pos_two_body(num_atomtypes*num_atomtypes),
three_body_descriptors(num_atomtypes*num_atomtypes*num_atomtypes),
pos_three_body(num_atomtypes*num_atomtypes*num_atomtypes),
max_cutoff(num_atomtypes*num_atomtypes)
{
  this->num_atomtypes = num_atomtypes;
  num_atomtypes_sq = num_atomtypes*num_atomtypes;
  global_max_cutoff = 0.0;
  printf("Constructor called with %d atom types\n",num_atomtypes);
}

DescriptorSet::~DescriptorSet(){}

void DescriptorSet::add_two_body_descriptor(
  int type1, int type2, int funtype, int num_prms, double* prms,
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun = switch_cutoff_functions(
    cutoff_type, cutoff);
  std::shared_ptr<TwoBodyDescriptor> symfun = switch_two_body_descriptors(
    funtype, num_prms, prms, cutfun);

  two_body_descriptors[type1*num_atomtypes+type2].push_back(symfun);
  num_descriptors[2*type1]++;
  for (int i = type2 + 1; i < num_atomtypes; i++)
  {
    pos_two_body[num_atomtypes*type1 + i]++;
  }

  if (cutoff > max_cutoff[type1*num_atomtypes+type2])
  {
    max_cutoff[type1*num_atomtypes+type2] = cutoff;
  }
  if (cutoff > global_max_cutoff)
  {
    global_max_cutoff = cutoff;
  }
}

void DescriptorSet::add_three_body_descriptor(
  int type1, int type2, int type3, int funtype, int num_prms, double* prms,
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun = switch_cutoff_functions(
    cutoff_type, cutoff);
  std::shared_ptr<ThreeBodyDescriptor> symfun = switch_three_body_descriptors(
    funtype, num_prms, prms, cutfun);
  // Atomtype2 and atomtype3 are sorted to maintain symmetry
  three_body_descriptors[num_atomtypes_sq*type1 + num_atomtypes*std::min(type2,type3) +
    std::max(type2,type3)].push_back(symfun);
  num_descriptors[2*type1 + 1]++;
  for (int j = type3 + 1; j < num_atomtypes; j++)
  {
    pos_three_body[num_atomtypes_sq*type1 + num_atomtypes*type2 + j]++;
  }
  for (int i = type2 + 1; i < num_atomtypes; i++)
  {
    for (int j = 0; j < num_atomtypes; j++)
    {
      pos_three_body[num_atomtypes_sq*type1 + num_atomtypes*i + j]++;
    }
  }
  if (cutoff > max_cutoff[type1*num_atomtypes + type2])
  {
    max_cutoff[type1*num_atomtypes + type2] = cutoff;
  }
  if (cutoff > max_cutoff[type1*num_atomtypes + type3])
  {
    max_cutoff[type1*num_atomtypes + type3] = cutoff;
  }
  if (cutoff > global_max_cutoff)
  {
    global_max_cutoff = cutoff;
  }
}

void DescriptorSet::print_descriptors() const
{
  printf("Number of atom types: %d\n", num_atomtypes);
  for (int ti = 0; ti < num_atomtypes; ti++)
  {
    printf("--- Atom type %d: ----\n", ti);
    printf("Number of two body descriptors for atom type %d is %d\n",
      ti, num_descriptors[2*ti]);
    printf("Number of three body descriptors for atom type %d is %d\n",
      ti, num_descriptors[2*ti+1]);
  }
}

int DescriptorSet::get_G_vector_size(int num_atoms, int* types)
{
  int G_vector_size = 0;
  for (int i = 0; i < num_atoms; i++)
  {
    G_vector_size += num_descriptors[2*types[i]] + num_descriptors[2*types[i]+1];
  }
  return G_vector_size;
}

void DescriptorSet::eval(
  int num_atoms, int* types, double* xyzs, double* G_vector)
{
  // Figure out the positions of symmetry functions centered on atom i and
  // save in pos_atoms
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i] + 1];
  }

  #pragma omp parallel
  {
    double rij, rik, rjk, costheta_i, costheta_j, costheta_k;
    int j, k, type_ij, type_ji, type_ijk, type_jki, type_kij;
    std::size_t two_Body_i, three_Body_i;
    // Actual evaluation of the symmetry functions. The sum over other atoms
    // is done for all symmetry functions simultaniously.
    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      // Loop over other atoms. To reduce computational effort symmetry functions
      // centered on atom j are also evaluated here. This allows to restrict the
      // loop to values j = i + 1.
      for (j = i + 1; j < num_atoms; j++)
      {
        rij = sqrt(pow(xyzs[3*i]-xyzs[3*j], 2) +
                  pow(xyzs[3*i + 1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i + 2]-xyzs[3*j+2], 2));
        type_ij = types[i]*num_atomtypes+types[j];
        type_ji = types[j]*num_atomtypes+types[i];

        if (rij < max_cutoff[type_ij] && rij < max_cutoff[type_ji])
        {
          // Add to two body symmetry functions centered on atom i
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ij].size();
            two_Body_i++)
          {
            #pragma omp atomic
            G_vector[pos_atoms[i] + pos_two_body[type_ij] + two_Body_i] +=
              two_body_descriptors[type_ij][two_Body_i]->eval(rij);
          }
          // Add to two body symmetry functions centered on atom j
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ji].size();
            two_Body_i++)
          {
            #pragma omp atomic
            G_vector[pos_atoms[j] + pos_two_body[type_ji] + two_Body_i] +=
              two_body_descriptors[type_ji][two_Body_i]->eval(rij);
          }
        }

        // Loop over third atom j:
        // TODO: maybe first check if even needed by checking number of
        // three_body_descriptors
        for (k = j + 1; k < num_atoms; k++)
        {
          rik = sqrt(pow(xyzs[3*i]-xyzs[3*k], 2) +
                    pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*k+2], 2));
          rjk = sqrt(pow(xyzs[3*j]-xyzs[3*k], 2) +
                    pow(xyzs[3*j+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*j+2]-xyzs[3*k+2], 2));
          if (rij > max_cutoff[type_ij] && rij > max_cutoff[type_ji] &&
            rik > max_cutoff[types[i]*num_atomtypes+types[k]] &&
            rik > max_cutoff[types[k]*num_atomtypes+types[i]] &&
            rjk > max_cutoff[types[j]*num_atomtypes+types[k]] &&
            rjk > max_cutoff[types[k]*num_atomtypes+types[j]])
          {
            continue;
          }
          // Calculate the angle between rij, rik and rjk
          costheta_i = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
            (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
            (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]))/(rij*rik);
          costheta_j = ((xyzs[3*j]-xyzs[3*k])*(xyzs[3*j]-xyzs[3*i]) +
            (xyzs[3*j+1]-xyzs[3*k+1])*(xyzs[3*j+1]-xyzs[3*i+1]) +
            (xyzs[3*j+2]-xyzs[3*k+2])*(xyzs[3*j+2]-xyzs[3*i+2]))/(rjk*rij);
          costheta_k = ((xyzs[3*k]-xyzs[3*i])*(xyzs[3*k]-xyzs[3*j]) +
            (xyzs[3*k+1]-xyzs[3*i+1])*(xyzs[3*k+1]-xyzs[3*j+1]) +
            (xyzs[3*k+2]-xyzs[3*i+2])*(xyzs[3*k+2]-xyzs[3*j+2]))/(rik*rjk);

          // As described in add_three_body_descriptor() the type of the three
          // body symmetry function consists of the atom type of the atom the
          // function is centered on an the sorted pair of atom types of the two
          // remaining atoms.

          // Add to three body symmetry functions centered on atom i.
          type_ijk = num_atomtypes_sq*types[i] +
            num_atomtypes*std::min(types[j], types[k]) +
            std::max(types[j], types[k]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_ijk].size();
            three_Body_i++)
          {
            #pragma omp atomic
            G_vector[pos_atoms[i] + num_descriptors[2*types[i]] +
              pos_three_body[type_ijk] + three_Body_i] +=
              three_body_descriptors[type_ijk][three_Body_i]->eval(rij, rik, costheta_i);
          }

          // Add to three body symmetry functions centered on atom j.
          type_jki = num_atomtypes_sq*types[j] +
            num_atomtypes*std::min(types[i],types[k]) +
            std::max(types[i],types[k]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_jki].size();
            three_Body_i++)
          {
            #pragma omp atomic
            G_vector[pos_atoms[j] + num_descriptors[2*types[j]] +
              pos_three_body[type_jki] + three_Body_i] +=
              three_body_descriptors[type_jki][three_Body_i]->eval(rij, rjk, costheta_j);
          }

          // Add to three body symmetry functions centered on atom k.
          type_kij = num_atomtypes_sq*types[k] +
            num_atomtypes*std::min(types[i],types[j]) +
            std::max(types[i],types[j]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_kij].size();
            three_Body_i++)
          {
            #pragma omp atomic
            G_vector[pos_atoms[k] + num_descriptors[2*types[k]] +
              pos_three_body[type_kij] + three_Body_i] +=
              three_body_descriptors[type_kij][three_Body_i]->eval(rjk, rik, costheta_k);
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}

void DescriptorSet::eval_derivatives(
  int num_atoms, int* types, double* xyzs, double* dG_tensor)
{
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i]+1];
  }

  #pragma omp parallel
  {
    double rij, rij2, rik, rik2, rjk, rjk2, costheta_i, costheta_j, costheta_k,
      dGdr, dGdrij, dGdrik, dGdcostheta, dot_i, dot_j, dot_k;
    int j, k, coord, index_base, type_ij, type_ji,
      type_ijk, type_jki, type_kij;
    std::size_t two_Body_i, three_Body_i;

    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      for (j = i + 1; j < num_atoms; j++)
      {

        rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
        rij = sqrt(rij2);

        type_ij = types[i]*num_atomtypes+types[j];
        type_ji = types[j]*num_atomtypes+types[i];
        if (rij < max_cutoff[type_ij] && rij < max_cutoff[type_ji])
        {
          // dG/dx is calculated as product of dG/dr * dr/dx
          // Add to two body symmetry functions centered on atom i
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ij].size();
            two_Body_i++)
          {
            dGdr = two_body_descriptors[type_ij][two_Body_i]->drij(rij);
            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] +
                two_Body_i) + 3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] +
                two_Body_i) + 3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
          // Add to two body symmetry functions centered on atom j
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ji].size();
            two_Body_i++)
          {
            dGdr = two_body_descriptors[type_ji][two_Body_i]->drij(rij);
            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[j] + pos_two_body[type_ji] + two_Body_i) +
                3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[j] + pos_two_body[type_ji] + two_Body_i) +
                3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
        }

        for (k = j + 1; k < num_atoms; k++)
        {
          rik2 = pow(xyzs[3*i]-xyzs[3*k], 2) + pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
          rik = sqrt(rik2);
          rjk2 = pow(xyzs[3*j]-xyzs[3*k], 2) + pow(xyzs[3*j+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*j+2]-xyzs[3*k+2], 2);
          rjk = sqrt(rjk2);
          if (rij > max_cutoff[type_ij] && rij > max_cutoff[type_ji] &&
            rik > max_cutoff[types[i]*num_atomtypes+types[k]] &&
            rik > max_cutoff[types[k]*num_atomtypes+types[i]] &&
            rjk > max_cutoff[types[j]*num_atomtypes+types[k]] &&
            rjk > max_cutoff[types[j]*num_atomtypes+types[k]])
          {
            continue;
          }
          dot_i = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
            (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
            (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
          dot_j = ((xyzs[3*j]-xyzs[3*k])*(xyzs[3*j]-xyzs[3*i]) +
            (xyzs[3*j+1]-xyzs[3*k+1])*(xyzs[3*j+1]-xyzs[3*i+1]) +
            (xyzs[3*j+2]-xyzs[3*k+2])*(xyzs[3*j+2]-xyzs[3*i+2]));
          dot_k = ((xyzs[3*k]-xyzs[3*i])*(xyzs[3*k]-xyzs[3*j]) +
            (xyzs[3*k+1]-xyzs[3*i+1])*(xyzs[3*k+1]-xyzs[3*j+1]) +
            (xyzs[3*k+2]-xyzs[3*i+2])*(xyzs[3*k+2]-xyzs[3*j+2]));
          // Calculate the angle between rij and rik
          costheta_i = (dot_i/(rij*rik));
          costheta_j = (dot_j/(rjk*rij));
          costheta_k = (dot_k/(rik*rjk));

          // Add to three body symmetry functions centered on atom i.
          type_ijk = num_atomtypes_sq*types[i] +
            num_atomtypes*std::min(types[j],types[k]) +
            std::max(types[j],types[k]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_ijk].size();
            three_Body_i++)
          {
            three_body_descriptors[type_ijk][three_Body_i]->derivatives(
              rij, rik, costheta_i, dGdrij, dGdrik, dGdcostheta);
            // Derivative with respect to costheta can fail for 0 or 180
            // degrees. TODO: deal with possible NaN (divide by zero) results.

            index_base = 3*num_atoms*(pos_atoms[i] + num_descriptors[2*types[i]] +
              pos_three_body[type_ijk]+ three_Body_i);
            for (coord = 0; coord < 3; coord++){
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrij*
                (xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrij*
                (-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrik*
                (xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrik*
                (-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(dot_i*rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) +
                dot_i*rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) +
                rij2*rik2*(xyzs[3*j + coord]+xyzs[3*k + coord] -
                2*xyzs[3*i + coord])) /
                (rij*rij2*rik*rik2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) -
                dot_i*(xyzs[3*i + coord]-xyzs[3*j + coord]))/
                (rij*rij2*rik + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) -
                dot_i*(xyzs[3*i + coord]-xyzs[3*k + coord]))/
                (rij*rik*rik2 + std::numeric_limits<double>::epsilon());
            }
          }

          // Add to three body symmetry functions centered on atom j.
          type_jki = num_atomtypes_sq*types[j] +
            num_atomtypes*std::min(types[k],types[i]) +
            std::max(types[k],types[i]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_jki].size();
            three_Body_i++)
          {
            three_body_descriptors[type_jki][three_Body_i]->derivatives(
              rjk, rij, costheta_j, dGdrij, dGdrik, dGdcostheta);

            index_base = 3*num_atoms*(pos_atoms[j] + num_descriptors[2*types[j]] +
              pos_three_body[type_jki]+ three_Body_i);
            for (coord = 0; coord < 3; coord++)
            {
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrij*
                (xyzs[3*j+coord]-xyzs[3*k+coord])/rjk;
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrij*
                (-xyzs[3*j+coord]+xyzs[3*k+coord])/rjk;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrik*
                (xyzs[3*j+coord]-xyzs[3*i+coord])/rij;
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrik*
                (-xyzs[3*j+coord]+xyzs[3*i+coord])/rij;

              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(dot_j*rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) +
                dot_j*rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) +
                rjk2*rij2*(xyzs[3*k + coord]+xyzs[3*i + coord] -
                2*xyzs[3*j + coord])) /
                (rjk*rjk2*rij*rij2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) -
                dot_j*(xyzs[3*j + coord]-xyzs[3*k + coord]))/
                (rjk*rjk2*rij + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) -
                dot_j*(xyzs[3*j + coord]-xyzs[3*i + coord]))/
                (rjk*rij*rij2 + std::numeric_limits<double>::epsilon());
            }
          }

          // Add to three body symmetry functions centered on atom k.
          type_kij = num_atomtypes_sq*types[k] +
            num_atomtypes*std::min(types[i],types[j]) +
            std::max(types[i],types[j]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_kij].size();
            three_Body_i++)
          {
            three_body_descriptors[type_kij][three_Body_i]->derivatives(
              rik, rjk, costheta_k, dGdrij, dGdrik, dGdcostheta);

            index_base = 3*num_atoms*(pos_atoms[k] + num_descriptors[2*types[k]] +
              pos_three_body[type_kij] + three_Body_i);
            for (coord = 0; coord < 3; coord++)
            {
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrij*
                (xyzs[3*k+coord]-xyzs[3*i+coord])/rik;
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrij*
                (-xyzs[3*k+coord]+xyzs[3*i+coord])/rik;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrik*
                (xyzs[3*k+coord]-xyzs[3*j+coord])/rjk;
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrik*
                (-xyzs[3*k+coord]+xyzs[3*j+coord])/rjk;


              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(dot_k*rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) +
                dot_k*rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) +
                rik2*rjk2*(xyzs[3*i + coord]+xyzs[3*j + coord] -
                2*xyzs[3*k + coord])) /
                (rik*rik2*rjk*rjk2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) -
                dot_k*(xyzs[3*k + coord]-xyzs[3*i + coord]))/
                (rik*rik2*rjk + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) -
                dot_k*(xyzs[3*k + coord]-xyzs[3*j + coord]))/
                (rik*rjk*rjk2 + std::numeric_limits<double>::epsilon());
            }
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}

void DescriptorSet::eval_with_derivatives(
  int num_atoms, int* types, double* xyzs, double* G_vector, double* dG_tensor)
{
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i]+1];
  }

  #pragma omp parallel
  {
    double rij, rij2, rik, rik2, rjk, rjk2, costheta_i, costheta_j, costheta_k,
      G, dGdr, dGdrij, dGdrik, dGdcostheta, dot_i, dot_j, dot_k;
    int j, k, coord, index_base, type_ij, type_ji, type_ijk, type_jki, type_kij;
    std::size_t two_Body_i, three_Body_i;

    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      for (j = i + 1; j < num_atoms; j++)
      {

        rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
        rij = sqrt(rij2);

        type_ij = types[i]*num_atomtypes+types[j];
        type_ji = types[j]*num_atomtypes+types[i];
        if (rij < max_cutoff[type_ij] && rij < max_cutoff[type_ji])
        {
          // Add to two body symmetry functions centered on atom i
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ij].size();
            two_Body_i++)
          {
            two_body_descriptors[type_ij][two_Body_i]->eval_with_derivatives(
              rij, G, dGdr);
            #pragma omp atomic
            G_vector[pos_atoms[i] + pos_two_body[type_ij] + two_Body_i] += G;

            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] +
                two_Body_i) + 3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] + two_Body_i) +
                3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
          // Add to two body symmetry functions centered on atom j
          for (two_Body_i = 0; two_Body_i < two_body_descriptors[type_ji].size();
            two_Body_i++)
          {
            two_body_descriptors[type_ji][two_Body_i]->eval_with_derivatives(
              rij, G, dGdr);
            #pragma omp atomic
            G_vector[pos_atoms[j] + pos_two_body[type_ji] + two_Body_i] += G;

            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[j] + pos_two_body[type_ji] + two_Body_i) +
                3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[3*num_atoms*(pos_atoms[j] + pos_two_body[type_ji] + two_Body_i) +
                3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
        }

        for (k = j + 1; k < num_atoms; k++)
        {
          rik2 = pow(xyzs[3*i]-xyzs[3*k], 2) + pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
          rik = sqrt(rik2);
          rjk2 = pow(xyzs[3*j]-xyzs[3*k], 2) + pow(xyzs[3*j+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*j+2]-xyzs[3*k+2], 2);
          rjk = sqrt(rjk2);
          if (rij > max_cutoff[type_ij] && rij > max_cutoff[type_ji] &&
            rik > max_cutoff[types[i]*num_atomtypes+types[k]] &&
            rik > max_cutoff[types[k]*num_atomtypes+types[i]] &&
            rjk > max_cutoff[types[j]*num_atomtypes+types[k]] &&
            rjk > max_cutoff[types[k]*num_atomtypes+types[j]])
          {
            continue;
          }
          dot_i = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
            (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
            (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
          dot_j = ((xyzs[3*j]-xyzs[3*k])*(xyzs[3*j]-xyzs[3*i]) +
            (xyzs[3*j+1]-xyzs[3*k+1])*(xyzs[3*j+1]-xyzs[3*i+1]) +
            (xyzs[3*j+2]-xyzs[3*k+2])*(xyzs[3*j+2]-xyzs[3*i+2]));
          dot_k = ((xyzs[3*k]-xyzs[3*i])*(xyzs[3*k]-xyzs[3*j]) +
            (xyzs[3*k+1]-xyzs[3*i+1])*(xyzs[3*k+1]-xyzs[3*j+1]) +
            (xyzs[3*k+2]-xyzs[3*i+2])*(xyzs[3*k+2]-xyzs[3*j+2]));
          // Calculate the angle between rij and rik
          costheta_i = (dot_i/(rij*rik));
          costheta_j = (dot_j/(rjk*rij));
          costheta_k = (dot_k/(rik*rjk));

          // As described in add_three_body_descriptor() the type of the three
          // body symmetry function consists of the atom type of the atom the
          // function is centered on an the sorted pair of atom types of the two
          // remaining atoms.

          // Add to three body symmetry functions centered on atom i.
          type_ijk = num_atomtypes_sq*types[i] +
            num_atomtypes*std::min(types[j], types[k]) +
            std::max(types[j], types[k]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_ijk].size();
            three_Body_i++)
          {
            three_body_descriptors[type_ijk][three_Body_i]->eval_with_derivatives(
              rij, rik, costheta_i, G, dGdrij, dGdrik, dGdcostheta);
            // Derivative with respect to costheta can fail for 0 or 180
            // degrees. TODO: deal with possible NaN (divide by zero) results.
            #pragma omp atomic
            G_vector[pos_atoms[i] + num_descriptors[2*types[i]] +
              pos_three_body[type_ijk] + three_Body_i] += G;

            index_base = 3*num_atoms*(pos_atoms[i] + num_descriptors[2*types[i]] +
              pos_three_body[type_ijk]+ three_Body_i);
            for (coord = 0; coord < 3; coord++)
            {
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrij*
                (xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrij*
                (-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrik*
                (xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrik*
                (-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(dot_i*rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) +
                dot_i*rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) +
                rij2*rik2*(xyzs[3*j + coord]+xyzs[3*k + coord] -
                2*xyzs[3*i + coord])) /
                (rij*rij2*rik*rik2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(rij2*(xyzs[3*i + coord]-xyzs[3*k + coord]) -
                dot_i*(xyzs[3*i + coord]-xyzs[3*j + coord]))/
                (rij*rij2*rik + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(rik2*(xyzs[3*i + coord]-xyzs[3*j + coord]) -
                dot_i*(xyzs[3*i + coord]-xyzs[3*k + coord]))/
                (rij*rik*rik2 + std::numeric_limits<double>::epsilon());
            }
          }

          // Add to three body symmetry functions centered on atom j.
          type_jki = num_atomtypes_sq*types[j] +
            num_atomtypes*std::min(types[i],types[k]) +
            std::max(types[i],types[k]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_jki].size();
            three_Body_i++)
          {
            three_body_descriptors[type_jki][three_Body_i]->eval_with_derivatives(
              rjk, rij, costheta_j, G, dGdrij, dGdrik, dGdcostheta);
            #pragma omp atomic
            G_vector[pos_atoms[j] + num_descriptors[2*types[j]] +
              pos_three_body[type_jki] + three_Body_i] += G;

            index_base = 3*num_atoms*(pos_atoms[j] + num_descriptors[2*types[j]] +
              pos_three_body[type_jki]+ three_Body_i);
            for (coord = 0; coord < 3; coord++)
            {
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrij*
                (xyzs[3*j+coord]-xyzs[3*k+coord])/rjk;
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrij*
                (-xyzs[3*j+coord]+xyzs[3*k+coord])/rjk;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrik*
                (xyzs[3*j+coord]-xyzs[3*i+coord])/rij;
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrik*
                (-xyzs[3*j+coord]+xyzs[3*i+coord])/rij;

              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(dot_j*rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) +
                dot_j*rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) +
                rjk2*rij2*(xyzs[3*k + coord]+xyzs[3*i + coord] -
                2*xyzs[3*j + coord])) /
                (rjk*rjk2*rij*rij2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(rjk2*(xyzs[3*j + coord]-xyzs[3*i + coord]) -
                dot_j*(xyzs[3*j + coord]-xyzs[3*k + coord]))/
                (rjk*rjk2*rij + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(rij2*(xyzs[3*j + coord]-xyzs[3*k + coord]) -
                dot_j*(xyzs[3*j + coord]-xyzs[3*i + coord]))/
                (rjk*rij*rij2 + std::numeric_limits<double>::epsilon());
            }
          }

          // Add to three body symmetry functions centered on atom k.
          type_kij = num_atomtypes_sq*types[k] +
            num_atomtypes*std::min(types[i],types[j]) +
            std::max(types[i],types[j]);
          for (three_Body_i = 0; three_Body_i < three_body_descriptors[type_kij].size();
            three_Body_i++)
          {
            three_body_descriptors[type_kij][three_Body_i]->eval_with_derivatives(
              rik, rjk, costheta_k, G, dGdrij, dGdrik, dGdcostheta);
            #pragma omp atomic
            G_vector[pos_atoms[k] + num_descriptors[2*types[k]] +
              pos_three_body[type_kij] + three_Body_i] += G;

            index_base = 3*num_atoms*(pos_atoms[k] + num_descriptors[2*types[k]] +
              pos_three_body[type_kij] + three_Body_i);
            for (coord = 0; coord < 3; coord++)
            {
              // Derivative with respect to rij
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrij*
                (xyzs[3*k+coord]-xyzs[3*i+coord])/rik;
              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdrij*
                (-xyzs[3*k+coord]+xyzs[3*i+coord])/rik;
              // Derivative with respect to rik
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdrik*
                (xyzs[3*k+coord]-xyzs[3*j+coord])/rjk;
              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdrik*
                (-xyzs[3*k+coord]+xyzs[3*j+coord])/rjk;


              // Derivative with respect to theta
              #pragma omp atomic
              dG_tensor[index_base + 3*k + coord] += dGdcostheta*
                -(dot_k*rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) +
                dot_k*rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) +
                rik2*rjk2*(xyzs[3*i + coord]+xyzs[3*j + coord] -
                2*xyzs[3*k + coord])) /
                (rik*rik2*rjk*rjk2 + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*i + coord] += dGdcostheta*
                -(rik2*(xyzs[3*k + coord]-xyzs[3*j + coord]) -
                dot_k*(xyzs[3*k + coord]-xyzs[3*i + coord]))/
                (rik*rik2*rjk + std::numeric_limits<double>::epsilon());

              #pragma omp atomic
              dG_tensor[index_base + 3*j + coord] += dGdcostheta*
                -(rjk2*(xyzs[3*k + coord]-xyzs[3*i + coord]) -
                dot_k*(xyzs[3*k + coord]-xyzs[3*j + coord]))/
                (rik*rjk*rjk2 + std::numeric_limits<double>::epsilon());
            }
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}

/* Evaluates the DescriptorSet for the given atomic geometry.
   Loops over every interation independently which makes it easier to execute
   in parallel.*/
void DescriptorSet::eval_atomwise(
  int num_atoms, int* types, double* xyzs, double* G_vector)
{
  // Figure out the positions of symmetry functions centered on atom i and
  // save in pos_atoms
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i] + 1];
  }

  #pragma omp parallel
  {
    double rij, rik, costheta;
    int j, k, type_ij, type_ijk;
    std::size_t two_Body_i, three_Body_i;

    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      for (j = 0; j < num_atoms; j++)
      {
        if (i == j)
        {
          continue;
        }
        rij = sqrt(pow(xyzs[3*i]-xyzs[3*j], 2) +
                  pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                  pow(xyzs[3*i+2]-xyzs[3*j+2], 2));
        // Combine the atomic indices i and j
        type_ij = types[i]*num_atomtypes+types[j];

        if (rij > max_cutoff[type_ij])
        {
          continue;
        }
        for (two_Body_i = 0;
          two_Body_i < two_body_descriptors[type_ij].size();
          two_Body_i++)
        {
          G_vector[pos_atoms[i] + pos_two_body[type_ij] + two_Body_i] +=
            two_body_descriptors[type_ij][two_Body_i]->eval(rij);
        }
        for (k = 0; k < num_atoms; k++)
        {
          if (i == k || j == k)
          {
            continue;
          }
          rik = sqrt(pow(xyzs[3*i]-xyzs[3*k], 2) +
                    pow(xyzs[3*i+1]-xyzs[3*k+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*k+2], 2));
          if (rik > max_cutoff[types[i]*num_atomtypes+types[k]])
          {
            continue;
          }

          // Combines the atomic indices i, j, and k
          type_ijk = num_atomtypes_sq*types[i]+num_atomtypes*types[j]+types[k];

          // Calculate the cosine of the angle between rij and rik
          costheta = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
            (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
            (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]))/(rij*rik);

          for (three_Body_i = 0;
            three_Body_i < three_body_descriptors[type_ijk].size(); three_Body_i++)
          {
            if (types[j] == types[k])
            {
              G_vector[pos_atoms[i] + num_descriptors[2*types[i]]
                + pos_three_body[type_ijk]+ three_Body_i] += 0.5*
                three_body_descriptors[type_ijk][three_Body_i]->eval(rij, rik, costheta);
            } else
            {
              G_vector[pos_atoms[i] + num_descriptors[2*types[i]]
                + pos_three_body[type_ijk]+ three_Body_i] +=
                three_body_descriptors[type_ijk][three_Body_i]->eval(rij, rik, costheta);
            }
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}

void DescriptorSet::eval_derivatives_atomwise(
  int num_atoms, int* types, double* xyzs, double* dG_tensor)
{
  // Figure out the positions of symmetry functions centered on atom i and
  // save in pos_atoms
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i] + 1];
  }

  #pragma omp parallel
  {
    double rij, rij2, rik, rik2, costheta, dGdr, dGdrij, dGdrik, dGdcostheta, dot;
    int j, k, coord, index_base, type_ij, type_ijk;
    std::size_t two_Body_i, three_Body_i;

    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      for (j = 0; j < num_atoms; j++)
      {
        if (i == j)
        {
          continue;
        } else
        {
          rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
          rij = sqrt(rij2);
          // Combine the atomic indices i and j
          type_ij = types[i]*num_atomtypes+types[j];

          // Loop over all two body descriptors
          for (two_Body_i = 0;
            two_Body_i < two_body_descriptors[type_ij].size();
            two_Body_i++)
          {
            dGdr = two_body_descriptors[type_ij][two_Body_i]->drij(rij);
            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              // dG/dx is calculated as product of dG/dr * dr/dx
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] + two_Body_i) +
                3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] + two_Body_i) +
                3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
          // Loop over the third atom k
          for (k = 0; k < num_atoms; k++)
          {
            if (i == k || j == k)
            {
              continue;
            } else
            {
              rik2 = pow(xyzs[3*i]-xyzs[3*k], 2)
                + pow(xyzs[3*i+1]-xyzs[3*k+1], 2)
                + pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
              rik = sqrt(rik2);

              if (rik > max_cutoff[types[i]*num_atomtypes+types[k]])
              {
                continue;
              }

              // Combines the atomic indices i, j, and k
              type_ijk =
                num_atomtypes_sq*types[i]+num_atomtypes*types[j]+types[k];

              dot = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
                (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
                (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
              costheta = (dot/(rij*rik));

              for (three_Body_i = 0;
                three_Body_i < three_body_descriptors[type_ijk].size();
                three_Body_i++)
              {
                three_body_descriptors[type_ijk][three_Body_i]->derivatives(
                  rij, rik, costheta, dGdrij, dGdrik, dGdcostheta);
                // Derivative with respect to costheta can fail for 0 or 180
                // degrees. TODO: deal with possible NaN (divide by zero) results.
                if (types[j] == types[k])
                {
                  dGdrij *= 0.5;
                  dGdrik *= 0.5;
                  dGdcostheta *= 0.5;
                }
                index_base = 3*num_atoms*(pos_atoms[i] + num_descriptors[2*types[i]]
                  + pos_three_body[type_ijk]+ three_Body_i);
                // Loop over the cartesian coordinates
                for (coord = 0; coord < 3; coord++){
                  // Derivative with respect to rij
                  // Add on atom i
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdrij*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
                  // Add on atom j
                  dG_tensor[index_base + 3*j + coord] +=
                    dGdrij*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
                  // Derivative with respect to rik
                  // Add on atom i
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdrik*(xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
                  // Add on atom j
                  dG_tensor[index_base + 3*k + coord] +=
                    dGdrik*(-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

                  // Derivative with respect to theta
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdcostheta*-(dot*rik2*(xyzs[3*i + coord]-xyzs[3*j+coord])
                    + dot*rij2*(xyzs[3*i + coord] - xyzs[3*k + coord])
                    + rij2*rik2*(xyzs[3*j + coord] + xyzs[3*k + coord] -
                      2*xyzs[3*i + coord]))/
                    (rij*rij2*rik*rik2+std::numeric_limits<double>::epsilon());

                  dG_tensor[index_base + 3*j + coord] +=
                    dGdcostheta*-(rij2*(xyzs[3*i+coord]-xyzs[3*k+coord])
                    - dot*(xyzs[3*i + coord]-xyzs[3*j+coord]))/
                    (rij*rij2*rik+std::numeric_limits<double>::epsilon());

                  dG_tensor[index_base + 3*k + coord] +=
                    dGdcostheta*-(rik2*(xyzs[3*i + coord]-xyzs[3*j+coord])
                    - dot*(xyzs[3*i+coord]-xyzs[3*k+coord]))/
                    (rij*rik*rik2+std::numeric_limits<double>::epsilon());
                }
              }
            }
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}

void DescriptorSet::eval_with_derivatives_atomwise(
  int num_atoms, int* types, double* xyzs, double* G_vector, double* dG_tensor)
{
  // Figure out the positions of symmetry functions centered on atom i and
  // save in pos_atoms
  int i, counter = 0;
  int* pos_atoms = new int[num_atoms];

  for (i = 0; i < num_atoms; i++)
  {
    pos_atoms[i] = counter;
    counter += num_descriptors[2*types[i]] + num_descriptors[2*types[i] + 1];
  }

  #pragma omp parallel
  {
    double rij, rij2, rik, rik2, costheta, G, dGdr, dGdrij, dGdrik,
      dGdcostheta, dot;
    int j, k, coord, index_base, type_ij, type_ijk;
    std::size_t two_Body_i, three_Body_i;

    #pragma omp for
    for (i = 0; i < num_atoms; i++)
    {
      for (j = 0; j < num_atoms; j++)
      {
        if (i == j)
        {
          continue;
        } else
        {
          rij2 = pow(xyzs[3*i]-xyzs[3*j], 2) + pow(xyzs[3*i+1]-xyzs[3*j+1], 2) +
                    pow(xyzs[3*i+2]-xyzs[3*j+2], 2);
          rij = sqrt(rij2);
          // Combine the atomic indices i and j
          type_ij = types[i]*num_atomtypes+types[j];

          // Loop over all two body descriptors
          for (two_Body_i = 0;
            two_Body_i < two_body_descriptors[type_ij].size();
            two_Body_i++)
          {
            two_body_descriptors[type_ij][two_Body_i]->eval_with_derivatives(rij, G, dGdr);
            G_vector[pos_atoms[i] + pos_two_body[type_ij] + two_Body_i] += G;
            // Loop over the three cartesian coordinates
            for (coord = 0; coord < 3; coord++){
              // dG/dx is calculated as product of dG/dr * dr/dx
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] + two_Body_i) +
                3*i+coord] += dGdr*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
              dG_tensor[3*num_atoms*(pos_atoms[i] + pos_two_body[type_ij] + two_Body_i) +
                3*j+coord] += dGdr*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
            }
          }
          // Loop over the third atom k
          for (k = 0; k < num_atoms; k++)
          {
            if (i == k || j == k)
            {
              continue;
            } else
            {
              rik2 = pow(xyzs[3*i]-xyzs[3*k], 2)
                + pow(xyzs[3*i+1]-xyzs[3*k+1], 2)
                + pow(xyzs[3*i+2]-xyzs[3*k+2], 2);
              rik = sqrt(rik2);

              if (rik > max_cutoff[types[i]*num_atomtypes+types[k]])
              {
                continue;
              }

              // Combines the atomic indices i, j, and k
              type_ijk =
                num_atomtypes_sq*types[i]+num_atomtypes*types[j]+types[k];

              dot = ((xyzs[3*i]-xyzs[3*j])*(xyzs[3*i]-xyzs[3*k]) +
                (xyzs[3*i+1]-xyzs[3*j+1])*(xyzs[3*i+1]-xyzs[3*k+1]) +
                (xyzs[3*i+2]-xyzs[3*j+2])*(xyzs[3*i+2]-xyzs[3*k+2]));
              costheta = (dot/(rij*rik));

              for (three_Body_i = 0;
                three_Body_i < three_body_descriptors[type_ijk].size();
                three_Body_i++)
              {
                three_body_descriptors[type_ijk][three_Body_i]->eval_with_derivatives(
                  rij, rik, costheta, G, dGdrij, dGdrik, dGdcostheta);
                // Derivative with respect to costheta can fail for 0 or 180
                // degrees. TODO: deal with possible NaN (divide by zero) results.
                if (types[j] == types[k])
                {
                  G *= 0.5;
                  dGdrij *= 0.5;
                  dGdrik *= 0.5;
                  dGdcostheta *= 0.5;
                }

                G_vector[pos_atoms[i] + num_descriptors[2*types[i]]
                  + pos_three_body[type_ijk]+ three_Body_i] += G;
                index_base = 3*num_atoms*(pos_atoms[i] + num_descriptors[2*types[i]]
                  + pos_three_body[type_ijk] + three_Body_i);
                // Loop over the cartesian coordinates
                for (coord = 0; coord < 3; coord++){
                  // Derivative with respect to rij
                  // Add on atom i
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdrij*(xyzs[3*i+coord]-xyzs[3*j+coord])/rij;
                  // Add on atom j
                  dG_tensor[index_base + 3*j + coord] +=
                    dGdrij*(-xyzs[3*i+coord]+xyzs[3*j+coord])/rij;
                  // Derivative with respect to rik
                  // Add on atom i
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdrik*(xyzs[3*i+coord]-xyzs[3*k+coord])/rik;
                  // Add on atom j
                  dG_tensor[index_base + 3*k + coord] +=
                    dGdrik*(-xyzs[3*i+coord]+xyzs[3*k+coord])/rik;

                  // Derivative with respect to theta
                  dG_tensor[index_base + 3*i + coord] +=
                    dGdcostheta*-(dot*rik2*(xyzs[3*i + coord]-xyzs[3*j+coord])
                    + dot*rij2*(xyzs[3*i + coord] - xyzs[3*k + coord])
                    + rij2*rik2*(xyzs[3*j + coord] + xyzs[3*k + coord] -
                      2*xyzs[3*i + coord]))/
                    (rij*rij2*rik*rik2+std::numeric_limits<double>::epsilon());

                  dG_tensor[index_base + 3*j + coord] +=
                    dGdcostheta*-(rij2*(xyzs[3*i+coord]-xyzs[3*k+coord])
                    - dot*(xyzs[3*i + coord]-xyzs[3*j+coord]))/
                    (rij*rij2*rik+std::numeric_limits<double>::epsilon());

                  dG_tensor[index_base + 3*k + coord] +=
                    dGdcostheta*-(rik2*(xyzs[3*i + coord]-xyzs[3*j+coord])
                    - dot*(xyzs[3*i+coord]-xyzs[3*k+coord]))/
                    (rij*rik*rik2+std::numeric_limits<double>::epsilon());
                }
              }
            }
          }
        }
      }
    }
  }
  delete[] pos_atoms;
}
