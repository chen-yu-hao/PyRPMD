/* Minimal real C ABI example: one hydrogen in a Cartesian harmonic potential.
 * V = 0.5 sum xyz^2, F = -xyz, atomic units. For API tests, not chemistry.
 */
#include "pyrpmd/pes_api_v1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

struct pyrpmd_pes_context_v1 { int initialized; };
static const char META[]="{\"schema\":\"pyrpmd.pes/1\",\"abi_version\":1,\"provider_name\":\"harmonic-example\",\"provider_version\":\"1\",\"artifact_sha256\":{},\"natoms\":1,\"atom_ids\":[\"0\"],\"atomic_numbers\":[1],\"state\":{\"label\":\"test\",\"charge\":0,\"multiplicity\":1},\"available_states\":[{\"label\":\"test\",\"charge\":0,\"multiplicity\":1}],\"layout\":\"B,N,3:C\",\"dtype\":\"float64\",\"position_unit\":\"bohr\",\"energy_unit\":\"hartree\",\"force_unit\":\"hartree/bohr\",\"boundary_conditions\":\"nonperiodic\",\"capabilities\":{\"native_batch\":true,\"force_kind\":\"analytic\",\"thread_safety\":\"reentrant\",\"stateful\":false,\"requires_bead_affinity\":false,\"execution_device\":\"cpu\"}}";
static void message(pyrpmd_error_v1* e,const char* s){if(!e)return;e->required_capacity=(int64_t)strlen(s)+1;if(e->message&&e->capacity>0)snprintf(e->message,(size_t)e->capacity,"%s",s);}
int32_t pyrpmd_pes_abi_version_v1(void){return 1;}
pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1* opts,pyrpmd_pes_context_v1** out,pyrpmd_error_v1* e){
 if(!out)return PYRPMD_INVALID_ARGUMENT_V1;*out=NULL;
 if(!opts||opts->abi_version!=1||opts->struct_size!=(int32_t)sizeof(*opts))return PYRPMD_ABI_MISMATCH_V1;
 *out=malloc(sizeof(**out));if(!*out)return PYRPMD_INTERNAL_ERROR_V1;(*out)->initialized=1;message(e,"");return 0;
}
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1* ctx,char* out,int64_t capacity,int64_t* required,pyrpmd_error_v1* e){
 if(!ctx||!required||capacity<0)return PYRPMD_INVALID_ARGUMENT_V1;
 *required=(int64_t)sizeof(META);if(capacity<*required){if(out&&capacity)out[0]=0;return PYRPMD_BUFFER_TOO_SMALL_V1;}
 if(!out)return PYRPMD_INVALID_ARGUMENT_V1;memcpy(out,META,sizeof(META));message(e,"");return 0;
}
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1* ctx,const pyrpmd_pes_input_v1* x,pyrpmd_pes_output_v1* y,pyrpmd_error_v1* e){
 if(!ctx||!x||!y)return PYRPMD_INVALID_ARGUMENT_V1;
 if(x->abi_version!=1||y->abi_version!=1||x->struct_size!=(int32_t)sizeof(*x)||y->struct_size!=(int32_t)sizeof(*y))return PYRPMD_ABI_MISMATCH_V1;
 int64_t b=x->nstructures;if(b<1||b>INT64_MAX/3||x->natoms!=1||x->layout!=1||x->scalar_type!=1||x->positions_count!=3*b||y->energies_count!=b||y->forces_count!=3*b||y->structure_status_count!=b||!x->positions||!y->energies||!y->forces||!y->structure_status)return PYRPMD_INVALID_ARGUMENT_V1;
 if(y->structure_messages && (y->structure_message_stride<1||b>INT64_MAX/y->structure_message_stride||y->structure_messages_capacity<b*y->structure_message_stride))return PYRPMD_INVALID_ARGUMENT_V1;
 for(int64_t i=0;i<b;i++){
  y->energies[i]=0.;y->structure_status[i]=0;
  for(int c=0;c<3;c++){double q=x->positions[3*i+c];y->energies[i]+=.5*q*q;y->forces[3*i+c]=-q;if(!isfinite(q))y->structure_status[i]=PYRPMD_STRUCTURE_NONFINITE_V1;}
  if(y->structure_status[i]){y->energies[i]=NAN;for(int c=0;c<3;c++)y->forces[3*i+c]=NAN;}
  if(y->structure_messages)y->structure_messages[i*y->structure_message_stride]=0;
 }
 message(e,"");return 0;
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1* ctx){free(ctx);}
