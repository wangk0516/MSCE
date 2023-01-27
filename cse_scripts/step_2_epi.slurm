#!/bin/bash
#SBATCH --account ctk_purchased
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH --output=epi_vasp.out
#SBATCH -o epi_vasp.o%j             # Name of stdout output file
#SBATCH -e epi_vasp.e%j             # Name of stderr error file
#SBATCH --partition=parallel

vasp_exec=/home/kw7rr/vasp5/vasp.5.4.4/bin/vasp_std

for i in `ls | grep stretch_`;
do
  cd $i
  pwd

  for j in `ls | grep POSCAR. | sed "s/POSCAR.//g" | sort -g`;
  do

    if ! [[ -e energy.$j ]]; then

      echo "calculating stretch $j ..."

      cp ../../INCAR.static INCAR
      cp ../../KPOINTS.static KPOINTS
      cp ../../POTCAR .

      cat POSCAR.$j > POSCAR
      srun $vasp_exec
      grep F= OSZICAR | awk '{print $5}' > energy.$j
      mv OSZICAR OSZICAR.$j

      rm INCAR KPOINTS POTCAR REPORT OUTCAR CH* WAVECAR vasprun.xml IB* DOS* CONTCAR EIGENVAL PCDAT XDATCAR

    fi

  done

  cd ..
done

