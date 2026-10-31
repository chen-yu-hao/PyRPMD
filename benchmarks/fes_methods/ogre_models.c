/* Independent implementation of the analytic models in Borgmans et al.,
 * JCTC 19 (2023) 9032, doi:10.1021/acs.jctc.3c01028, equations 13 and 14.
 * Build with -DACKLEY for the second model. Numerical CVs map to bohr here;
 * energy formulas are in kJ/mol. The auxiliary z mode is separable.
 * Ackley has a common quartic wall outside |x|,|y|=8, outside the paper's
 * comparison domain [-6,6]^2, to define a normalizable target ensemble.
 */
#include "pyrpmd/pes_api_v1.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct pyrpmd_pes_context_v1 { int initialized; };
#ifdef ACKLEY
#define MODEL_NAME "ogre-ackley-with-exterior-wall"
#else
#define MODEL_NAME "ogre-double-well"
#endif
static const char META[]="{\"schema\":\"pyrpmd.pes/1\",\"abi_version\":1,\"provider_name\":\"" MODEL_NAME "\",\"provider_version\":\"1\",\"artifact_sha256\":{},\"natoms\":1,\"atom_ids\":[\"0\"],\"atomic_numbers\":[1],\"state\":{\"label\":\"analytic-test\",\"charge\":0,\"multiplicity\":1},\"available_states\":[{\"label\":\"analytic-test\",\"charge\":0,\"multiplicity\":1}],\"layout\":\"B,N,3:C\",\"dtype\":\"float64\",\"position_unit\":\"bohr\",\"energy_unit\":\"hartree\",\"force_unit\":\"hartree/bohr\",\"boundary_conditions\":\"nonperiodic\",\"capabilities\":{\"native_batch\":true,\"force_kind\":\"analytic\",\"thread_safety\":\"reentrant\",\"stateful\":false,\"requires_bead_affinity\":false,\"execution_device\":\"cpu\"}}";
static void msg(pyrpmd_error_v1 *e,const char *s){if(!e)return;e->required_capacity=strlen(s)+1;if(e->message&&e->capacity>0)snprintf(e->message,e->capacity,"%s",s);}
int32_t pyrpmd_pes_abi_version_v1(void){return 1;}
pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1 *o,pyrpmd_pes_context_v1 **out,pyrpmd_error_v1 *e){
 if(!out)return PYRPMD_INVALID_ARGUMENT_V1;*out=NULL;
 if(!o||o->abi_version!=1||o->struct_size!=sizeof(*o))return PYRPMD_ABI_MISMATCH_V1;
 *out=calloc(1,sizeof(**out));if(!*out)return PYRPMD_INTERNAL_ERROR_V1;msg(e,"");return 0;
}
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1 *ctx,char *out,int64_t cap,int64_t *need,pyrpmd_error_v1 *e){
 if(!ctx||!need||cap<0)return PYRPMD_INVALID_ARGUMENT_V1;*need=sizeof(META);
 if(cap<*need)return PYRPMD_BUFFER_TOO_SMALL_V1;if(!out)return PYRPMD_INVALID_ARGUMENT_V1;
 memcpy(out,META,sizeof(META));msg(e,"");return 0;
}
static void model(const double *q,double *energy,double *force){
 double x=q[0],y=q[1],z=q[2],u,gx,gy;
#ifdef ACKLEY
 const double pi=3.14159265358979323846;
 double r=sqrt(0.5*(x*x+y*y)), a=exp(-0.2*r), b=exp(0.5*(cos(2*pi*x)+cos(2*pi*y)));
 u=-20*a-b+exp(1.0)+20;
 gx=(r>0?2*a*x/r:0)+pi*b*sin(2*pi*x);
 gy=(r>0?2*a*y/r:0)+pi*b*sin(2*pi*y);
 double dx=fmax(fabs(x)-8,0),dy=fmax(fabs(y)-8,0);
 u+=4*(dx*dx*dx*dx+dy*dy*dy*dy);
 gx+=copysign(16*dx*dx*dx,x);gy+=copysign(16*dy*dy*dy,y);
#else
 double r2=x*x+y*y;
 u=r2*r2;gx=4*r2*x;gy=4*r2*y;
 for(int s=-1;s<=1;s+=2){
  double dx=x-s*0.2,dy=y-s*0.4,v=-10*exp(-30*dx*dx-3*dy*dy);
  u+=v;gx+=v*(-60*dx);gy+=v*(-6*dy);
 }
#endif
 const double kjmol_per_hartree=2625.4996394799;
 *energy=(u+5*z*z)/kjmol_per_hartree;
 force[0]=-gx/kjmol_per_hartree;force[1]=-gy/kjmol_per_hartree;force[2]=-10*z/kjmol_per_hartree;
}
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1 *ctx,const pyrpmd_pes_input_v1 *x,pyrpmd_pes_output_v1 *y,pyrpmd_error_v1 *e){
 if(!ctx||!x||!y)return PYRPMD_INVALID_ARGUMENT_V1;
 if(x->abi_version!=1||y->abi_version!=1||x->struct_size!=sizeof(*x)||y->struct_size!=sizeof(*y))return PYRPMD_ABI_MISMATCH_V1;
 int64_t b=x->nstructures;
 if(b<1||b>INT64_MAX/3||x->natoms!=1||x->layout!=1||x->scalar_type!=1||x->positions_count!=3*b||y->energies_count!=b||y->forces_count!=3*b||y->structure_status_count!=b||!x->positions||!y->energies||!y->forces||!y->structure_status)return PYRPMD_INVALID_ARGUMENT_V1;
 if(y->structure_messages&&(y->structure_message_stride<1||b>INT64_MAX/y->structure_message_stride||y->structure_messages_capacity<b*y->structure_message_stride))return PYRPMD_INVALID_ARGUMENT_V1;
 for(int64_t i=0;i<b;i++){
  model(x->positions+3*i,y->energies+i,y->forces+3*i);y->structure_status[i]=0;
  if(!isfinite(y->energies[i]))y->structure_status[i]=PYRPMD_STRUCTURE_NONFINITE_V1;
  for(int j=0;j<3;j++)if(!isfinite(y->forces[3*i+j]))y->structure_status[i]=PYRPMD_STRUCTURE_NONFINITE_V1;
  if(y->structure_messages)y->structure_messages[i*y->structure_message_stride]=0;
 }
 msg(e,"");return 0;
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1 *ctx){free(ctx);}
