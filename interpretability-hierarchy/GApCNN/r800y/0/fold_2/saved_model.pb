��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
�
Adam/dense_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dense_layer_1/bias/v
�
-Adam/dense_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/dense_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nameAdam/dense_layer_1/kernel/v
�
/Adam/dense_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer_1/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/dense_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam/dense_layer_0/bias/v
�
-Adam/dense_layer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer_0/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameAdam/dense_layer_0/kernel/v
�
/Adam/dense_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_layer_0/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/conv_layer_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_4/bias/v
�
,Adam/conv_layer_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_4/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/conv_layer_4/kernel/v
�
.Adam/conv_layer_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_4/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv_layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_3/bias/v
�
,Adam/conv_layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/conv_layer_3/kernel/v
�
.Adam/conv_layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_3/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_2/bias/v
�
,Adam/conv_layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/conv_layer_2/kernel/v
�
.Adam/conv_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_2/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/conv_layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv_layer_1/bias/v
�
,Adam/conv_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv_layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv_layer_1/kernel/v
�
.Adam/conv_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_1/kernel/v*&
_output_shapes
: @*
dtype0
�
Adam/conv_layer_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv_layer_0/bias/v
�
,Adam/conv_layer_0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_0/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv_layer_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv_layer_0/kernel/v
�
.Adam/conv_layer_0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv_layer_0/kernel/v*&
_output_shapes
: *
dtype0
�
Adam/dense_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/dense_layer_1/bias/m
�
-Adam/dense_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_nameAdam/dense_layer_1/kernel/m
�
/Adam/dense_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer_1/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/dense_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameAdam/dense_layer_0/bias/m
�
-Adam/dense_layer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer_0/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*,
shared_nameAdam/dense_layer_0/kernel/m
�
/Adam/dense_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_layer_0/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/conv_layer_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_4/bias/m
�
,Adam/conv_layer_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_4/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/conv_layer_4/kernel/m
�
.Adam/conv_layer_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_4/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv_layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_3/bias/m
�
,Adam/conv_layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*+
shared_nameAdam/conv_layer_3/kernel/m
�
.Adam/conv_layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_3/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_nameAdam/conv_layer_2/bias/m
�
,Adam/conv_layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*+
shared_nameAdam/conv_layer_2/kernel/m
�
.Adam/conv_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_2/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/conv_layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv_layer_1/bias/m
�
,Adam/conv_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv_layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameAdam/conv_layer_1/kernel/m
�
.Adam/conv_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_1/kernel/m*&
_output_shapes
: @*
dtype0
�
Adam/conv_layer_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv_layer_0/bias/m
�
,Adam/conv_layer_0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_0/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv_layer_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/conv_layer_0/kernel/m
�
.Adam/conv_layer_0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv_layer_0/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
|
dense_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedense_layer_1/bias
u
&dense_layer_1/bias/Read/ReadVariableOpReadVariableOpdense_layer_1/bias*
_output_shapes
:*
dtype0
�
dense_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*%
shared_namedense_layer_1/kernel
~
(dense_layer_1/kernel/Read/ReadVariableOpReadVariableOpdense_layer_1/kernel*
_output_shapes
:	�*
dtype0
}
dense_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*#
shared_namedense_layer_0/bias
v
&dense_layer_0/bias/Read/ReadVariableOpReadVariableOpdense_layer_0/bias*
_output_shapes	
:�*
dtype0
�
dense_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*%
shared_namedense_layer_0/kernel

(dense_layer_0/kernel/Read/ReadVariableOpReadVariableOpdense_layer_0/kernel* 
_output_shapes
:
��*
dtype0
{
conv_layer_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameconv_layer_4/bias
t
%conv_layer_4/bias/Read/ReadVariableOpReadVariableOpconv_layer_4/bias*
_output_shapes	
:�*
dtype0
�
conv_layer_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameconv_layer_4/kernel
�
'conv_layer_4/kernel/Read/ReadVariableOpReadVariableOpconv_layer_4/kernel*(
_output_shapes
:��*
dtype0
{
conv_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameconv_layer_3/bias
t
%conv_layer_3/bias/Read/ReadVariableOpReadVariableOpconv_layer_3/bias*
_output_shapes	
:�*
dtype0
�
conv_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*$
shared_nameconv_layer_3/kernel
�
'conv_layer_3/kernel/Read/ReadVariableOpReadVariableOpconv_layer_3/kernel*(
_output_shapes
:��*
dtype0
{
conv_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameconv_layer_2/bias
t
%conv_layer_2/bias/Read/ReadVariableOpReadVariableOpconv_layer_2/bias*
_output_shapes	
:�*
dtype0
�
conv_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*$
shared_nameconv_layer_2/kernel
�
'conv_layer_2/kernel/Read/ReadVariableOpReadVariableOpconv_layer_2/kernel*'
_output_shapes
:@�*
dtype0
z
conv_layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv_layer_1/bias
s
%conv_layer_1/bias/Read/ReadVariableOpReadVariableOpconv_layer_1/bias*
_output_shapes
:@*
dtype0
�
conv_layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*$
shared_nameconv_layer_1/kernel
�
'conv_layer_1/kernel/Read/ReadVariableOpReadVariableOpconv_layer_1/kernel*&
_output_shapes
: @*
dtype0
z
conv_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv_layer_0/bias
s
%conv_layer_0/bias/Read/ReadVariableOpReadVariableOpconv_layer_0/bias*
_output_shapes
: *
dtype0
�
conv_layer_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameconv_layer_0/kernel
�
'conv_layer_0/kernel/Read/ReadVariableOpReadVariableOpconv_layer_0/kernel*&
_output_shapes
: *
dtype0
|
gaussian_model/sigmaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namegaussian_model/sigma
u
(gaussian_model/sigma/Read/ReadVariableOpReadVariableOpgaussian_model/sigma*
_output_shapes
: *
dtype0
�
gaussian_model/projVarHandleOp*
_output_shapes
: *
dtype0*
shape:_�*$
shared_namegaussian_model/proj
�
'gaussian_model/proj/Read/ReadVariableOpReadVariableOpgaussian_model/proj*#
_output_shapes
:_�*
dtype0
�
serving_default_inputPlaceholder*0
_output_shapes
:���������_�*
dtype0*%
shape:���������_�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputgaussian_model/projgaussian_model/sigmaconv_layer_0/kernelconv_layer_0/biasconv_layer_1/kernelconv_layer_1/biasconv_layer_2/kernelconv_layer_2/biasconv_layer_3/kernelconv_layer_3/biasconv_layer_4/kernelconv_layer_4/biasdense_layer_0/kerneldense_layer_0/biasdense_layer_1/kerneldense_layer_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_signature_wrapper_96909

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� Bݖ
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
proj
	sigma*
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
 layer_with_weights-4
 layer-9
!layer-10
"layer-11
#layer_with_weights-5
#layer-12
$layer-13
%layer_with_weights-6
%layer-14
&layer-15
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 
z
0
1
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15*
j
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
* 
�

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate
Xiter9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�Cm�Dm�Em�Fm�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�Cv�Dv�Ev�Fv�*

Yserving_default* 

0
1*
* 
* 
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
a[
VARIABLE_VALUEgaussian_model/proj4layer_with_weights-0/proj/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEgaussian_model/sigma5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

9kernel
:bias
 g_jit_compiled_convolution_op*
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses* 
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

;kernel
<bias
 t_jit_compiled_convolution_op*
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses* 
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Akernel
Bbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ekernel
Fbias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
j
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13*
j
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13*
:
�0
�1
�2
�3
�4
�5
�6* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
SM
VARIABLE_VALUEconv_layer_0/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_0/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEconv_layer_4/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEconv_layer_4/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_layer_0/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_layer_0/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_layer_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_layer_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*

0
1*
'
0
1
2
3
4*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
* 
* 
* 
* 
* 
* 

90
:1*

90
:1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

;0
<1*

;0
<1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

=0
>1*

=0
>1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

?0
@1*

?0
@1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

A0
B1*

A0
B1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

C0
D1*

C0
D1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

E0
F1*

E0
F1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 

�trace_0* 
* 
z
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 


�0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
vp
VARIABLE_VALUEAdam/conv_layer_0/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_0/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_1/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_2/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_2/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_3/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_3/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_layer_4/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_layer_4/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_0/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_0/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_1/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_1/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_0/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_0/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_1/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_2/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_2/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_3/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_3/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/conv_layer_4/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/conv_layer_4/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_0/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_0/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_1/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_1/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'gaussian_model/proj/Read/ReadVariableOp(gaussian_model/sigma/Read/ReadVariableOp'conv_layer_0/kernel/Read/ReadVariableOp%conv_layer_0/bias/Read/ReadVariableOp'conv_layer_1/kernel/Read/ReadVariableOp%conv_layer_1/bias/Read/ReadVariableOp'conv_layer_2/kernel/Read/ReadVariableOp%conv_layer_2/bias/Read/ReadVariableOp'conv_layer_3/kernel/Read/ReadVariableOp%conv_layer_3/bias/Read/ReadVariableOp'conv_layer_4/kernel/Read/ReadVariableOp%conv_layer_4/bias/Read/ReadVariableOp(dense_layer_0/kernel/Read/ReadVariableOp&dense_layer_0/bias/Read/ReadVariableOp(dense_layer_1/kernel/Read/ReadVariableOp&dense_layer_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/conv_layer_0/kernel/m/Read/ReadVariableOp,Adam/conv_layer_0/bias/m/Read/ReadVariableOp.Adam/conv_layer_1/kernel/m/Read/ReadVariableOp,Adam/conv_layer_1/bias/m/Read/ReadVariableOp.Adam/conv_layer_2/kernel/m/Read/ReadVariableOp,Adam/conv_layer_2/bias/m/Read/ReadVariableOp.Adam/conv_layer_3/kernel/m/Read/ReadVariableOp,Adam/conv_layer_3/bias/m/Read/ReadVariableOp.Adam/conv_layer_4/kernel/m/Read/ReadVariableOp,Adam/conv_layer_4/bias/m/Read/ReadVariableOp/Adam/dense_layer_0/kernel/m/Read/ReadVariableOp-Adam/dense_layer_0/bias/m/Read/ReadVariableOp/Adam/dense_layer_1/kernel/m/Read/ReadVariableOp-Adam/dense_layer_1/bias/m/Read/ReadVariableOp.Adam/conv_layer_0/kernel/v/Read/ReadVariableOp,Adam/conv_layer_0/bias/v/Read/ReadVariableOp.Adam/conv_layer_1/kernel/v/Read/ReadVariableOp,Adam/conv_layer_1/bias/v/Read/ReadVariableOp.Adam/conv_layer_2/kernel/v/Read/ReadVariableOp,Adam/conv_layer_2/bias/v/Read/ReadVariableOp.Adam/conv_layer_3/kernel/v/Read/ReadVariableOp,Adam/conv_layer_3/bias/v/Read/ReadVariableOp.Adam/conv_layer_4/kernel/v/Read/ReadVariableOp,Adam/conv_layer_4/bias/v/Read/ReadVariableOp/Adam/dense_layer_0/kernel/v/Read/ReadVariableOp-Adam/dense_layer_0/bias/v/Read/ReadVariableOp/Adam/dense_layer_1/kernel/v/Read/ReadVariableOp-Adam/dense_layer_1/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *'
f"R 
__inference__traced_save_98096
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegaussian_model/projgaussian_model/sigmaconv_layer_0/kernelconv_layer_0/biasconv_layer_1/kernelconv_layer_1/biasconv_layer_2/kernelconv_layer_2/biasconv_layer_3/kernelconv_layer_3/biasconv_layer_4/kernelconv_layer_4/biasdense_layer_0/kerneldense_layer_0/biasdense_layer_1/kerneldense_layer_1/biasbeta_1beta_2decaylearning_rate	Adam/itertotal_3count_3total_2count_2total_1count_1totalcountAdam/conv_layer_0/kernel/mAdam/conv_layer_0/bias/mAdam/conv_layer_1/kernel/mAdam/conv_layer_1/bias/mAdam/conv_layer_2/kernel/mAdam/conv_layer_2/bias/mAdam/conv_layer_3/kernel/mAdam/conv_layer_3/bias/mAdam/conv_layer_4/kernel/mAdam/conv_layer_4/bias/mAdam/dense_layer_0/kernel/mAdam/dense_layer_0/bias/mAdam/dense_layer_1/kernel/mAdam/dense_layer_1/bias/mAdam/conv_layer_0/kernel/vAdam/conv_layer_0/bias/vAdam/conv_layer_1/kernel/vAdam/conv_layer_1/bias/vAdam/conv_layer_2/kernel/vAdam/conv_layer_2/bias/vAdam/conv_layer_3/kernel/vAdam/conv_layer_3/bias/vAdam/conv_layer_4/kernel/vAdam/conv_layer_4/bias/vAdam/dense_layer_0/kernel/vAdam/dense_layer_0/bias/vAdam/dense_layer_1/kernel/vAdam/dense_layer_1/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� **
f%R#
!__inference__traced_restore_98277�
�	
�
__inference_loss_fn_5_97893S
?dense_layer_0_kernel_regularizer_l2loss_readvariableop_resource:
��
identity��6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp?dense_layer_0_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: f
IdentityIdentity(dense_layer_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
,__inference_conv_layer_1_layer_call_fn_97640

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������G@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������/� : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_97768

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95774a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_conv_layer_0_layer_call_fn_97607

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������/� `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_95889	
input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_95858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�l
�

@__inference_model_layer_call_and_return_conditional_losses_96256	
input,
conv_layer_0_96184:  
conv_layer_0_96186: ,
conv_layer_1_96190: @ 
conv_layer_1_96192:@-
conv_layer_2_96196:@�!
conv_layer_2_96198:	�.
conv_layer_3_96202:��!
conv_layer_3_96204:	�.
conv_layer_4_96208:��!
conv_layer_4_96210:	�'
dense_layer_0_96215:
��"
dense_layer_0_96217:	�&
dense_layer_1_96221:	�!
dense_layer_1_96223:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputconv_layer_0_96184conv_layer_0_96186*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647�
!conv_activation_0/PartitionedCallPartitionedCall-conv_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_96190conv_layer_1_96192*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674�
!conv_activation_1/PartitionedCallPartitionedCall-conv_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_96196conv_layer_2_96198*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701�
!conv_activation_2/PartitionedCallPartitionedCall-conv_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_96202conv_layer_3_96204*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728�
!conv_activation_3/PartitionedCallPartitionedCall-conv_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_96208conv_layer_4_96210*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755�
!conv_activation_4/PartitionedCallPartitionedCall-conv_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766�
flatten/PartitionedCallPartitionedCall*conv_activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95774�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_96215dense_layer_0_96217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790�
"dense_activation_0/PartitionedCallPartitionedCall.dense_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_96221dense_layer_1_96223*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817�
"dense_activation_1/PartitionedCallPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_96184*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_96190*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_96196*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_96202*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_96208*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_96215* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_96221*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^conv_layer_0/StatefulPartitionedCall6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_1/StatefulPartitionedCall6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_2/StatefulPartitionedCall6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_3/StatefulPartitionedCall6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_4/StatefulPartitionedCall6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_0/StatefulPartitionedCall7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_1/StatefulPartitionedCall7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2L
$conv_layer_0/StatefulPartitionedCall$conv_layer_0/StatefulPartitionedCall2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_1/StatefulPartitionedCall$conv_layer_1/StatefulPartitionedCall2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_2/StatefulPartitionedCall$conv_layer_2/StatefulPartitionedCall2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_3/StatefulPartitionedCall$conv_layer_3/StatefulPartitionedCall2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_4/StatefulPartitionedCall$conv_layer_4/StatefulPartitionedCall2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_0/StatefulPartitionedCall%dense_layer_0/StatefulPartitionedCall2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�	
�
__inference_loss_fn_0_97848X
>conv_layer_0_kernel_regularizer_l2loss_readvariableop_resource: 
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>conv_layer_0_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'conv_layer_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
2__inference_perturbative_model_layer_call_fn_96498	
input
unknown:_�
	unknown_0: #
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�
M
1__inference_conv_activation_2_layer_call_fn_97692

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������#�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������#�:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�!
�
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374
x8
!tensordot_readvariableop_resource:_�%
mul_readvariableop_resource: 
identity��Tensordot/ReadVariableOp�mul/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*#
_output_shapes
:_�*
dtype0c
Tensordot/axesConst*
_output_shapes
:*
dtype0*!
valueB"         X
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: @
Tensordot/ShapeShapex*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposexTensordot/concat:output:0*
T0*0
_output_shapes
:���������_��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"��     �
Tensordot/Reshape_1Reshape Tensordot/ReadVariableOp:value:0"Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������T
Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:{
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*#
_output_shapes
:���������Q
ones_like/ShapeShapeTensordot:output:0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:���������f
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0h
mulMulones_like:output:0mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:���������z
stackPackTensordot:output:0mul:z:0*
N*
T0*'
_output_shapes
:���������*
axis���������]
IdentityIdentitystack:output:0^NoOp*
T0*'
_output_shapes
:���������v
NoOpNoOp^Tensordot/ReadVariableOp^mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:S O
0
_output_shapes
:���������_�

_user_specified_namex
�
�
%__inference_model_layer_call_fn_97369

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_95858o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
�
2__inference_perturbative_model_layer_call_fn_96700	
input
unknown:_�
	unknown_0: #
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96628o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�
�
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_97753

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_97654

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������G@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������/� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
�
i
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_97807

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_3_97875Z
>conv_layer_3_kernel_regularizer_l2loss_readvariableop_resource:��
identity��5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>conv_layer_3_kernel_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'conv_layer_3/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
,__inference_conv_layer_4_layer_call_fn_97739

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
G
0__inference_Sigma_Activation_layer_call_fn_97583
x
identity�
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������#�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_97774

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
,__inference_conv_layer_3_layer_call_fn_97706

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������#�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96768	
input+
gaussian_model_96703:_�
gaussian_model_96705: %
model_96708: 
model_96710: %
model_96712: @
model_96714:@&
model_96716:@�
model_96718:	�'
model_96720:��
model_96722:	�'
model_96724:��
model_96726:	�
model_96728:
��
model_96730:	�
model_96732:	�
model_96734:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&gaussian_model/StatefulPartitionedCall�model/StatefulPartitionedCall�
&gaussian_model/StatefulPartitionedCallStatefulPartitionedCallinputgaussian_model_96703gaussian_model_96705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374�
model/StatefulPartitionedCallStatefulPartitionedCallinputmodel_96708model_96710model_96712model_96714model_96716model_96718model_96720model_96722model_96724model_96726model_96728model_96730model_96732model_96734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_95858�
add/PartitionedCallPartitionedCall/gaussian_model/StatefulPartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_96415�
 Sigma_Activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96708*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96712*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96716*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96720*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96724*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96728* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96732*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)Sigma_Activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^gaussian_model/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&gaussian_model/StatefulPartitionedCall&gaussian_model/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�
h
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������G@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������G@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������G@:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
�
�
-__inference_dense_layer_1_layer_call_fn_97816

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_perturbative_model_layer_call_fn_97011

inputs
unknown:_�
	unknown_0: #
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96628o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
j
>__inference_add_layer_call_and_return_conditional_losses_97578
inputs_0
inputs_1
identityR
addAddV2inputs_0inputs_1*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
b
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432
x
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask^
SoftplusSoftplusstrided_slice:output:0*
T0*'
_output_shapes
:���������f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2strided_slice_1:output:0Softplus:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
.__inference_gaussian_model_layer_call_fn_97272
x
unknown:_�
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:���������_�

_user_specified_namex
�l
�

@__inference_model_layer_call_and_return_conditional_losses_96331	
input,
conv_layer_0_96259:  
conv_layer_0_96261: ,
conv_layer_1_96265: @ 
conv_layer_1_96267:@-
conv_layer_2_96271:@�!
conv_layer_2_96273:	�.
conv_layer_3_96277:��!
conv_layer_3_96279:	�.
conv_layer_4_96283:��!
conv_layer_4_96285:	�'
dense_layer_0_96290:
��"
dense_layer_0_96292:	�&
dense_layer_1_96296:	�!
dense_layer_1_96298:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputconv_layer_0_96259conv_layer_0_96261*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647�
!conv_activation_0/PartitionedCallPartitionedCall-conv_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_96265conv_layer_1_96267*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674�
!conv_activation_1/PartitionedCallPartitionedCall-conv_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_96271conv_layer_2_96273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701�
!conv_activation_2/PartitionedCallPartitionedCall-conv_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_96277conv_layer_3_96279*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728�
!conv_activation_3/PartitionedCallPartitionedCall-conv_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_96283conv_layer_4_96285*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755�
!conv_activation_4/PartitionedCallPartitionedCall-conv_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766�
flatten/PartitionedCallPartitionedCall*conv_activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95774�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_96290dense_layer_0_96292*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790�
"dense_activation_0/PartitionedCallPartitionedCall.dense_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_96296dense_layer_1_96298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817�
"dense_activation_1/PartitionedCallPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_96259*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_96265*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_96271*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_96277*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_96283*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_96290* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_96296*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^conv_layer_0/StatefulPartitionedCall6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_1/StatefulPartitionedCall6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_2/StatefulPartitionedCall6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_3/StatefulPartitionedCall6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_4/StatefulPartitionedCall6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_0/StatefulPartitionedCall7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_1/StatefulPartitionedCall7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2L
$conv_layer_0/StatefulPartitionedCall$conv_layer_0/StatefulPartitionedCall2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_1/StatefulPartitionedCall$conv_layer_1/StatefulPartitionedCall2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_2/StatefulPartitionedCall$conv_layer_2/StatefulPartitionedCall2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_3/StatefulPartitionedCall$conv_layer_3/StatefulPartitionedCall2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_4/StatefulPartitionedCall$conv_layer_4/StatefulPartitionedCall2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_0/StatefulPartitionedCall%dense_layer_0/StatefulPartitionedCall2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�	
�
__inference_loss_fn_6_97902R
?dense_layer_1_kernel_regularizer_l2loss_readvariableop_resource:	�
identity��6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp?dense_layer_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: f
IdentityIdentity(dense_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp
�
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_97839

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������G@�
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������/� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_4_97884Z
>conv_layer_4_kernel_regularizer_l2loss_readvariableop_resource:��
identity��5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>conv_layer_4_kernel_regularizer_l2loss_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'conv_layer_4/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp
�t
�
@__inference_model_layer_call_and_return_conditional_losses_97484

inputsE
+conv_layer_0_conv2d_readvariableop_resource: :
,conv_layer_0_biasadd_readvariableop_resource: E
+conv_layer_1_conv2d_readvariableop_resource: @:
,conv_layer_1_biasadd_readvariableop_resource:@F
+conv_layer_2_conv2d_readvariableop_resource:@�;
,conv_layer_2_biasadd_readvariableop_resource:	�G
+conv_layer_3_conv2d_readvariableop_resource:��;
,conv_layer_3_biasadd_readvariableop_resource:	�G
+conv_layer_4_conv2d_readvariableop_resource:��;
,conv_layer_4_biasadd_readvariableop_resource:	�@
,dense_layer_0_matmul_readvariableop_resource:
��<
-dense_layer_0_biasadd_readvariableop_resource:	�?
,dense_layer_1_matmul_readvariableop_resource:	�;
-dense_layer_1_biasadd_readvariableop_resource:
identity��#conv_layer_0/BiasAdd/ReadVariableOp�"conv_layer_0/Conv2D/ReadVariableOp�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_1/BiasAdd/ReadVariableOp�"conv_layer_1/Conv2D/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_2/BiasAdd/ReadVariableOp�"conv_layer_2/Conv2D/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_3/BiasAdd/ReadVariableOp�"conv_layer_3/Conv2D/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_4/BiasAdd/ReadVariableOp�"conv_layer_4/Conv2D/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�$dense_layer_0/BiasAdd/ReadVariableOp�#dense_layer_0/MatMul/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$dense_layer_1/BiasAdd/ReadVariableOp�#dense_layer_1/MatMul/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
"conv_layer_0/Conv2D/ReadVariableOpReadVariableOp+conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv_layer_0/Conv2DConv2Dinputs*conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
#conv_layer_0/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv_layer_0/BiasAddBiasAddconv_layer_0/Conv2D:output:0+conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� x
conv_activation_0/ReluReluconv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
"conv_layer_1/Conv2D/ReadVariableOpReadVariableOp+conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv_layer_1/Conv2DConv2D$conv_activation_0/Relu:activations:0*conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
#conv_layer_1/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_layer_1/BiasAddBiasAddconv_layer_1/Conv2D:output:0+conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@w
conv_activation_1/ReluReluconv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
"conv_layer_2/Conv2D/ReadVariableOpReadVariableOp+conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv_layer_2/Conv2DConv2D$conv_activation_1/Relu:activations:0*conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
#conv_layer_2/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_2/BiasAddBiasAddconv_layer_2/Conv2D:output:0+conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�x
conv_activation_2/ReluReluconv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
"conv_layer_3/Conv2D/ReadVariableOpReadVariableOp+conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_layer_3/Conv2DConv2D$conv_activation_2/Relu:activations:0*conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
#conv_layer_3/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_3/BiasAddBiasAddconv_layer_3/Conv2D:output:0+conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������x
conv_activation_3/ReluReluconv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
"conv_layer_4/Conv2D/ReadVariableOpReadVariableOp+conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_layer_4/Conv2DConv2D$conv_activation_3/Relu:activations:0*conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
#conv_layer_4/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_4/BiasAddBiasAddconv_layer_4/Conv2D:output:0+conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������x
conv_activation_4/ReluReluconv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape$conv_activation_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
#dense_layer_0/MatMul/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_layer_0/MatMulMatMulflatten/Reshape:output:0+dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_layer_0/BiasAddBiasAdddense_layer_0/MatMul:product:0,dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
dense_activation_0/ReluReludense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_layer_1/MatMulMatMul%dense_activation_0/Relu:activations:0+dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_layer_1/BiasAddBiasAdddense_layer_1/MatMul:product:0,dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: m
IdentityIdentitydense_layer_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^conv_layer_0/BiasAdd/ReadVariableOp#^conv_layer_0/Conv2D/ReadVariableOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_1/BiasAdd/ReadVariableOp#^conv_layer_1/Conv2D/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_2/BiasAdd/ReadVariableOp#^conv_layer_2/Conv2D/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_3/BiasAdd/ReadVariableOp#^conv_layer_3/Conv2D/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_4/BiasAdd/ReadVariableOp#^conv_layer_4/Conv2D/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp%^dense_layer_0/BiasAdd/ReadVariableOp$^dense_layer_0/MatMul/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^dense_layer_1/BiasAdd/ReadVariableOp$^dense_layer_1/MatMul/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2J
#conv_layer_0/BiasAdd/ReadVariableOp#conv_layer_0/BiasAdd/ReadVariableOp2H
"conv_layer_0/Conv2D/ReadVariableOp"conv_layer_0/Conv2D/ReadVariableOp2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_1/BiasAdd/ReadVariableOp#conv_layer_1/BiasAdd/ReadVariableOp2H
"conv_layer_1/Conv2D/ReadVariableOp"conv_layer_1/Conv2D/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_2/BiasAdd/ReadVariableOp#conv_layer_2/BiasAdd/ReadVariableOp2H
"conv_layer_2/Conv2D/ReadVariableOp"conv_layer_2/Conv2D/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_3/BiasAdd/ReadVariableOp#conv_layer_3/BiasAdd/ReadVariableOp2H
"conv_layer_3/Conv2D/ReadVariableOp"conv_layer_3/Conv2D/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_4/BiasAdd/ReadVariableOp#conv_layer_4/BiasAdd/ReadVariableOp2H
"conv_layer_4/Conv2D/ReadVariableOp"conv_layer_4/Conv2D/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2L
$dense_layer_0/BiasAdd/ReadVariableOp$dense_layer_0/BiasAdd/ReadVariableOp2J
#dense_layer_0/MatMul/ReadVariableOp#dense_layer_0/MatMul/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$dense_layer_1/BiasAdd/ReadVariableOp$dense_layer_1/BiasAdd/ReadVariableOp2J
#dense_layer_1/MatMul/ReadVariableOp#dense_layer_1/MatMul/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�t
�
@__inference_model_layer_call_and_return_conditional_losses_97566

inputsE
+conv_layer_0_conv2d_readvariableop_resource: :
,conv_layer_0_biasadd_readvariableop_resource: E
+conv_layer_1_conv2d_readvariableop_resource: @:
,conv_layer_1_biasadd_readvariableop_resource:@F
+conv_layer_2_conv2d_readvariableop_resource:@�;
,conv_layer_2_biasadd_readvariableop_resource:	�G
+conv_layer_3_conv2d_readvariableop_resource:��;
,conv_layer_3_biasadd_readvariableop_resource:	�G
+conv_layer_4_conv2d_readvariableop_resource:��;
,conv_layer_4_biasadd_readvariableop_resource:	�@
,dense_layer_0_matmul_readvariableop_resource:
��<
-dense_layer_0_biasadd_readvariableop_resource:	�?
,dense_layer_1_matmul_readvariableop_resource:	�;
-dense_layer_1_biasadd_readvariableop_resource:
identity��#conv_layer_0/BiasAdd/ReadVariableOp�"conv_layer_0/Conv2D/ReadVariableOp�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_1/BiasAdd/ReadVariableOp�"conv_layer_1/Conv2D/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_2/BiasAdd/ReadVariableOp�"conv_layer_2/Conv2D/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_3/BiasAdd/ReadVariableOp�"conv_layer_3/Conv2D/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�#conv_layer_4/BiasAdd/ReadVariableOp�"conv_layer_4/Conv2D/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�$dense_layer_0/BiasAdd/ReadVariableOp�#dense_layer_0/MatMul/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$dense_layer_1/BiasAdd/ReadVariableOp�#dense_layer_1/MatMul/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
"conv_layer_0/Conv2D/ReadVariableOpReadVariableOp+conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv_layer_0/Conv2DConv2Dinputs*conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
#conv_layer_0/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv_layer_0/BiasAddBiasAddconv_layer_0/Conv2D:output:0+conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� x
conv_activation_0/ReluReluconv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
"conv_layer_1/Conv2D/ReadVariableOpReadVariableOp+conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv_layer_1/Conv2DConv2D$conv_activation_0/Relu:activations:0*conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
#conv_layer_1/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv_layer_1/BiasAddBiasAddconv_layer_1/Conv2D:output:0+conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@w
conv_activation_1/ReluReluconv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
"conv_layer_2/Conv2D/ReadVariableOpReadVariableOp+conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv_layer_2/Conv2DConv2D$conv_activation_1/Relu:activations:0*conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
#conv_layer_2/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_2/BiasAddBiasAddconv_layer_2/Conv2D:output:0+conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�x
conv_activation_2/ReluReluconv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
"conv_layer_3/Conv2D/ReadVariableOpReadVariableOp+conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_layer_3/Conv2DConv2D$conv_activation_2/Relu:activations:0*conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
#conv_layer_3/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_3/BiasAddBiasAddconv_layer_3/Conv2D:output:0+conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������x
conv_activation_3/ReluReluconv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
"conv_layer_4/Conv2D/ReadVariableOpReadVariableOp+conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv_layer_4/Conv2DConv2D$conv_activation_3/Relu:activations:0*conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
#conv_layer_4/BiasAdd/ReadVariableOpReadVariableOp,conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv_layer_4/BiasAddBiasAddconv_layer_4/Conv2D:output:0+conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������x
conv_activation_4/ReluReluconv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshape$conv_activation_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:�����������
#dense_layer_0/MatMul/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_layer_0/MatMulMatMulflatten/Reshape:output:0+dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_layer_0/BiasAddBiasAdddense_layer_0/MatMul:product:0,dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������r
dense_activation_0/ReluReludense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
#dense_layer_1/MatMul/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
dense_layer_1/MatMulMatMul%dense_activation_0/Relu:activations:0+dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp-dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_layer_1/BiasAddBiasAdddense_layer_1/MatMul:product:0,dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp+conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: m
IdentityIdentitydense_layer_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^conv_layer_0/BiasAdd/ReadVariableOp#^conv_layer_0/Conv2D/ReadVariableOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_1/BiasAdd/ReadVariableOp#^conv_layer_1/Conv2D/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_2/BiasAdd/ReadVariableOp#^conv_layer_2/Conv2D/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_3/BiasAdd/ReadVariableOp#^conv_layer_3/Conv2D/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp$^conv_layer_4/BiasAdd/ReadVariableOp#^conv_layer_4/Conv2D/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp%^dense_layer_0/BiasAdd/ReadVariableOp$^dense_layer_0/MatMul/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^dense_layer_1/BiasAdd/ReadVariableOp$^dense_layer_1/MatMul/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2J
#conv_layer_0/BiasAdd/ReadVariableOp#conv_layer_0/BiasAdd/ReadVariableOp2H
"conv_layer_0/Conv2D/ReadVariableOp"conv_layer_0/Conv2D/ReadVariableOp2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_1/BiasAdd/ReadVariableOp#conv_layer_1/BiasAdd/ReadVariableOp2H
"conv_layer_1/Conv2D/ReadVariableOp"conv_layer_1/Conv2D/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_2/BiasAdd/ReadVariableOp#conv_layer_2/BiasAdd/ReadVariableOp2H
"conv_layer_2/Conv2D/ReadVariableOp"conv_layer_2/Conv2D/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_3/BiasAdd/ReadVariableOp#conv_layer_3/BiasAdd/ReadVariableOp2H
"conv_layer_3/Conv2D/ReadVariableOp"conv_layer_3/Conv2D/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2J
#conv_layer_4/BiasAdd/ReadVariableOp#conv_layer_4/BiasAdd/ReadVariableOp2H
"conv_layer_4/Conv2D/ReadVariableOp"conv_layer_4/Conv2D/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2L
$dense_layer_0/BiasAdd/ReadVariableOp$dense_layer_0/BiasAdd/ReadVariableOp2J
#dense_layer_0/MatMul/ReadVariableOp#dense_layer_0/MatMul/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$dense_layer_1/BiasAdd/ReadVariableOp$dense_layer_1/BiasAdd/ReadVariableOp2J
#dense_layer_1/MatMul/ReadVariableOp#dense_layer_1/MatMul/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������/� c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������/� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������/� :X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_1_97857X
>conv_layer_1_kernel_regularizer_l2loss_readvariableop_resource: @
identity��5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>conv_layer_1_kernel_regularizer_l2loss_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'conv_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp
�
h
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_97697

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������#�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������#�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������#�:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
M
1__inference_conv_activation_4_layer_call_fn_97758

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������#��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������G@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_97402

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_96117o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
O
#__inference_add_layer_call_fn_97572
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_96415`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_97830

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
>__inference_add_layer_call_and_return_conditional_losses_96415

inputs
inputs_1
identityP
addAddV2inputsinputs_1*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_95774

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
I__inference_gaussian_model_layer_call_and_return_conditional_losses_97308
x8
!tensordot_readvariableop_resource:_�%
mul_readvariableop_resource: 
identity��Tensordot/ReadVariableOp�mul/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*#
_output_shapes
:_�*
dtype0c
Tensordot/axesConst*
_output_shapes
:*
dtype0*!
valueB"         X
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: @
Tensordot/ShapeShapex*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposexTensordot/concat:output:0*
T0*0
_output_shapes
:���������_��
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"��     �
Tensordot/Reshape_1Reshape Tensordot/ReadVariableOp:value:0"Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������T
Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:{
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*#
_output_shapes
:���������Q
ones_like/ShapeShapeTensordot:output:0*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:���������f
mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
: *
dtype0h
mulMulones_like:output:0mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:���������z
stackPackTensordot:output:0mul:z:0*
N*
T0*'
_output_shapes
:���������*
axis���������]
IdentityIdentitystack:output:0^NoOp*
T0*'
_output_shapes
:���������v
NoOpNoOp^Tensordot/ReadVariableOp^mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2(
mul/ReadVariableOpmul/ReadVariableOp:S O
0
_output_shapes
:���������_�

_user_specified_namex
�
M
1__inference_conv_activation_1_layer_call_fn_97659

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������G@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������G@:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
�l
�

@__inference_model_layer_call_and_return_conditional_losses_96117

inputs,
conv_layer_0_96045:  
conv_layer_0_96047: ,
conv_layer_1_96051: @ 
conv_layer_1_96053:@-
conv_layer_2_96057:@�!
conv_layer_2_96059:	�.
conv_layer_3_96063:��!
conv_layer_3_96065:	�.
conv_layer_4_96069:��!
conv_layer_4_96071:	�'
dense_layer_0_96076:
��"
dense_layer_0_96078:	�&
dense_layer_1_96082:	�!
dense_layer_1_96084:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_0_96045conv_layer_0_96047*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647�
!conv_activation_0/PartitionedCallPartitionedCall-conv_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_96051conv_layer_1_96053*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674�
!conv_activation_1/PartitionedCallPartitionedCall-conv_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_96057conv_layer_2_96059*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701�
!conv_activation_2/PartitionedCallPartitionedCall-conv_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_96063conv_layer_3_96065*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728�
!conv_activation_3/PartitionedCallPartitionedCall-conv_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_96069conv_layer_4_96071*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755�
!conv_activation_4/PartitionedCallPartitionedCall-conv_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766�
flatten/PartitionedCallPartitionedCall*conv_activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95774�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_96076dense_layer_0_96078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790�
"dense_activation_0/PartitionedCallPartitionedCall.dense_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_96082dense_layer_1_96084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817�
"dense_activation_1/PartitionedCallPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_96045*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_96051*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_96057*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_96063*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_96069*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_96076* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_96082*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^conv_layer_0/StatefulPartitionedCall6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_1/StatefulPartitionedCall6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_2/StatefulPartitionedCall6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_3/StatefulPartitionedCall6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_4/StatefulPartitionedCall6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_0/StatefulPartitionedCall7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_1/StatefulPartitionedCall7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2L
$conv_layer_0/StatefulPartitionedCall$conv_layer_0/StatefulPartitionedCall2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_1/StatefulPartitionedCall$conv_layer_1/StatefulPartitionedCall2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_2/StatefulPartitionedCall$conv_layer_2/StatefulPartitionedCall2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_3/StatefulPartitionedCall$conv_layer_3/StatefulPartitionedCall2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_4/StatefulPartitionedCall$conv_layer_4/StatefulPartitionedCall2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_0/StatefulPartitionedCall%dense_layer_0/StatefulPartitionedCall2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_97763

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_97664

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������G@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������G@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������G@:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
�
�
2__inference_perturbative_model_layer_call_fn_96974

inputs
unknown:_�
	unknown_0: #
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96463o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�l
�

@__inference_model_layer_call_and_return_conditional_losses_95858

inputs,
conv_layer_0_95648:  
conv_layer_0_95650: ,
conv_layer_1_95675: @ 
conv_layer_1_95677:@-
conv_layer_2_95702:@�!
conv_layer_2_95704:	�.
conv_layer_3_95729:��!
conv_layer_3_95731:	�.
conv_layer_4_95756:��!
conv_layer_4_95758:	�'
dense_layer_0_95791:
��"
dense_layer_0_95793:	�&
dense_layer_1_95818:	�!
dense_layer_1_95820:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_0_95648conv_layer_0_95650*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647�
!conv_activation_0/PartitionedCallPartitionedCall-conv_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_95675conv_layer_1_95677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_95674�
!conv_activation_1/PartitionedCallPartitionedCall-conv_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_95685�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_95702conv_layer_2_95704*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701�
!conv_activation_2/PartitionedCallPartitionedCall-conv_layer_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_95729conv_layer_3_95731*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_95728�
!conv_activation_3/PartitionedCallPartitionedCall-conv_layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_95756conv_layer_4_95758*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_95755�
!conv_activation_4/PartitionedCallPartitionedCall-conv_layer_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_95766�
flatten/PartitionedCallPartitionedCall*conv_activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_95774�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_95791dense_layer_0_95793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790�
"dense_activation_0/PartitionedCallPartitionedCall.dense_layer_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_95818dense_layer_1_95820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_95817�
"dense_activation_1/PartitionedCallPartitionedCall.dense_layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_95648*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_95675*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_95702*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_95729*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_95756*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_95791* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_95818*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: z
IdentityIdentity+dense_activation_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp%^conv_layer_0/StatefulPartitionedCall6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_1/StatefulPartitionedCall6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_2/StatefulPartitionedCall6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_3/StatefulPartitionedCall6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp%^conv_layer_4/StatefulPartitionedCall6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_0/StatefulPartitionedCall7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp&^dense_layer_1/StatefulPartitionedCall7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2L
$conv_layer_0/StatefulPartitionedCall$conv_layer_0/StatefulPartitionedCall2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_1/StatefulPartitionedCall$conv_layer_1/StatefulPartitionedCall2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_2/StatefulPartitionedCall$conv_layer_2/StatefulPartitionedCall2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_3/StatefulPartitionedCall$conv_layer_3/StatefulPartitionedCall2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2L
$conv_layer_4/StatefulPartitionedCall$conv_layer_4/StatefulPartitionedCall2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_0/StatefulPartitionedCall%dense_layer_0/StatefulPartitionedCall2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2N
%dense_layer_1/StatefulPartitionedCall%dense_layer_1/StatefulPartitionedCall2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_97797

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_97687

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������#��
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������G@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
�
M
1__inference_conv_activation_0_layer_call_fn_97626

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_95658i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:���������/� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������/� :X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
��
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97263

inputsG
0gaussian_model_tensordot_readvariableop_resource:_�4
*gaussian_model_mul_readvariableop_resource: K
1model_conv_layer_0_conv2d_readvariableop_resource: @
2model_conv_layer_0_biasadd_readvariableop_resource: K
1model_conv_layer_1_conv2d_readvariableop_resource: @@
2model_conv_layer_1_biasadd_readvariableop_resource:@L
1model_conv_layer_2_conv2d_readvariableop_resource:@�A
2model_conv_layer_2_biasadd_readvariableop_resource:	�M
1model_conv_layer_3_conv2d_readvariableop_resource:��A
2model_conv_layer_3_biasadd_readvariableop_resource:	�M
1model_conv_layer_4_conv2d_readvariableop_resource:��A
2model_conv_layer_4_biasadd_readvariableop_resource:	�F
2model_dense_layer_0_matmul_readvariableop_resource:
��B
3model_dense_layer_0_biasadd_readvariableop_resource:	�E
2model_dense_layer_1_matmul_readvariableop_resource:	�A
3model_dense_layer_1_biasadd_readvariableop_resource:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�'gaussian_model/Tensordot/ReadVariableOp�!gaussian_model/mul/ReadVariableOp�)model/conv_layer_0/BiasAdd/ReadVariableOp�(model/conv_layer_0/Conv2D/ReadVariableOp�)model/conv_layer_1/BiasAdd/ReadVariableOp�(model/conv_layer_1/Conv2D/ReadVariableOp�)model/conv_layer_2/BiasAdd/ReadVariableOp�(model/conv_layer_2/Conv2D/ReadVariableOp�)model/conv_layer_3/BiasAdd/ReadVariableOp�(model/conv_layer_3/Conv2D/ReadVariableOp�)model/conv_layer_4/BiasAdd/ReadVariableOp�(model/conv_layer_4/Conv2D/ReadVariableOp�*model/dense_layer_0/BiasAdd/ReadVariableOp�)model/dense_layer_0/MatMul/ReadVariableOp�*model/dense_layer_1/BiasAdd/ReadVariableOp�)model/dense_layer_1/MatMul/ReadVariableOp�
'gaussian_model/Tensordot/ReadVariableOpReadVariableOp0gaussian_model_tensordot_readvariableop_resource*#
_output_shapes
:_�*
dtype0r
gaussian_model/Tensordot/axesConst*
_output_shapes
:*
dtype0*!
valueB"         g
gaussian_model/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: T
gaussian_model/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:h
&gaussian_model/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!gaussian_model/Tensordot/GatherV2GatherV2'gaussian_model/Tensordot/Shape:output:0&gaussian_model/Tensordot/free:output:0/gaussian_model/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(gaussian_model/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#gaussian_model/Tensordot/GatherV2_1GatherV2'gaussian_model/Tensordot/Shape:output:0&gaussian_model/Tensordot/axes:output:01gaussian_model/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
gaussian_model/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gaussian_model/Tensordot/ProdProd*gaussian_model/Tensordot/GatherV2:output:0'gaussian_model/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 gaussian_model/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gaussian_model/Tensordot/Prod_1Prod,gaussian_model/Tensordot/GatherV2_1:output:0)gaussian_model/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$gaussian_model/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
gaussian_model/Tensordot/concatConcatV2&gaussian_model/Tensordot/free:output:0&gaussian_model/Tensordot/axes:output:0-gaussian_model/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
gaussian_model/Tensordot/stackPack&gaussian_model/Tensordot/Prod:output:0(gaussian_model/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"gaussian_model/Tensordot/transpose	Transposeinputs(gaussian_model/Tensordot/concat:output:0*
T0*0
_output_shapes
:���������_��
 gaussian_model/Tensordot/ReshapeReshape&gaussian_model/Tensordot/transpose:y:0'gaussian_model/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������y
(gaussian_model/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"��     �
"gaussian_model/Tensordot/Reshape_1Reshape/gaussian_model/Tensordot/ReadVariableOp:value:01gaussian_model/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
gaussian_model/Tensordot/MatMulMatMul)gaussian_model/Tensordot/Reshape:output:0+gaussian_model/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������c
 gaussian_model/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB h
&gaussian_model/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!gaussian_model/Tensordot/concat_1ConcatV2*gaussian_model/Tensordot/GatherV2:output:0)gaussian_model/Tensordot/Const_2:output:0/gaussian_model/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
gaussian_model/TensordotReshape)gaussian_model/Tensordot/MatMul:product:0*gaussian_model/Tensordot/concat_1:output:0*
T0*#
_output_shapes
:���������o
gaussian_model/ones_like/ShapeShape!gaussian_model/Tensordot:output:0*
T0*
_output_shapes
:c
gaussian_model/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gaussian_model/ones_likeFill'gaussian_model/ones_like/Shape:output:0'gaussian_model/ones_like/Const:output:0*
T0*#
_output_shapes
:����������
!gaussian_model/mul/ReadVariableOpReadVariableOp*gaussian_model_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
gaussian_model/mulMul!gaussian_model/ones_like:output:0)gaussian_model/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:����������
gaussian_model/stackPack!gaussian_model/Tensordot:output:0gaussian_model/mul:z:0*
N*
T0*'
_output_shapes
:���������*
axis����������
(model/conv_layer_0/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv_layer_0/Conv2DConv2Dinputs0model/conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
)model/conv_layer_0/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv_layer_0/BiasAddBiasAdd"model/conv_layer_0/Conv2D:output:01model/conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
model/conv_activation_0/ReluRelu#model/conv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
(model/conv_layer_1/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
model/conv_layer_1/Conv2DConv2D*model/conv_activation_0/Relu:activations:00model/conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
)model/conv_layer_1/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv_layer_1/BiasAddBiasAdd"model/conv_layer_1/Conv2D:output:01model/conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
model/conv_activation_1/ReluRelu#model/conv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
(model/conv_layer_2/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
model/conv_layer_2/Conv2DConv2D*model/conv_activation_1/Relu:activations:00model/conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
)model/conv_layer_2/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_2/BiasAddBiasAdd"model/conv_layer_2/Conv2D:output:01model/conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
model/conv_activation_2/ReluRelu#model/conv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
(model/conv_layer_3/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv_layer_3/Conv2DConv2D*model/conv_activation_2/Relu:activations:00model/conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
)model/conv_layer_3/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_3/BiasAddBiasAdd"model/conv_layer_3/Conv2D:output:01model/conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
model/conv_activation_3/ReluRelu#model/conv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
(model/conv_layer_4/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv_layer_4/Conv2DConv2D*model/conv_activation_3/Relu:activations:00model/conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
)model/conv_layer_4/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_4/BiasAddBiasAdd"model/conv_layer_4/Conv2D:output:01model/conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
model/conv_activation_4/ReluRelu#model/conv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape*model/conv_activation_4/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
)model/dense_layer_0/MatMul/ReadVariableOpReadVariableOp2model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_layer_0/MatMulMatMulmodel/flatten/Reshape:output:01model/dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*model/dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp3model_dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_layer_0/BiasAddBiasAdd$model/dense_layer_0/MatMul:product:02model/dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
model/dense_activation_0/ReluRelu$model/dense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)model/dense_layer_1/MatMul/ReadVariableOpReadVariableOp2model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_layer_1/MatMulMatMul+model/dense_activation_0/Relu:activations:01model/dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model/dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp3model_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_layer_1/BiasAddBiasAdd$model/dense_layer_1/MatMul:product:02model/dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
add/addAddV2gaussian_model/stack:output:0$model/dense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������u
$Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����w
&Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Sigma_Activation/strided_sliceStridedSliceadd/add:z:0-Sigma_Activation/strided_slice/stack:output:0/Sigma_Activation/strided_slice/stack_1:output:0/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
Sigma_Activation/SoftplusSoftplus'Sigma_Activation/strided_slice:output:0*
T0*'
_output_shapes
:���������w
&Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����y
(Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 Sigma_Activation/strided_slice_1StridedSliceadd/add:z:0/Sigma_Activation/strided_slice_1/stack:output:01Sigma_Activation/strided_slice_1/stack_1:output:01Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskg
Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Sigma_Activation/concatConcatV2)Sigma_Activation/strided_slice_1:output:0'Sigma_Activation/Softplus:activations:0%Sigma_Activation/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp2model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp2model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: o
IdentityIdentity Sigma_Activation/concat:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp(^gaussian_model/Tensordot/ReadVariableOp"^gaussian_model/mul/ReadVariableOp*^model/conv_layer_0/BiasAdd/ReadVariableOp)^model/conv_layer_0/Conv2D/ReadVariableOp*^model/conv_layer_1/BiasAdd/ReadVariableOp)^model/conv_layer_1/Conv2D/ReadVariableOp*^model/conv_layer_2/BiasAdd/ReadVariableOp)^model/conv_layer_2/Conv2D/ReadVariableOp*^model/conv_layer_3/BiasAdd/ReadVariableOp)^model/conv_layer_3/Conv2D/ReadVariableOp*^model/conv_layer_4/BiasAdd/ReadVariableOp)^model/conv_layer_4/Conv2D/ReadVariableOp+^model/dense_layer_0/BiasAdd/ReadVariableOp*^model/dense_layer_0/MatMul/ReadVariableOp+^model/dense_layer_1/BiasAdd/ReadVariableOp*^model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2R
'gaussian_model/Tensordot/ReadVariableOp'gaussian_model/Tensordot/ReadVariableOp2F
!gaussian_model/mul/ReadVariableOp!gaussian_model/mul/ReadVariableOp2V
)model/conv_layer_0/BiasAdd/ReadVariableOp)model/conv_layer_0/BiasAdd/ReadVariableOp2T
(model/conv_layer_0/Conv2D/ReadVariableOp(model/conv_layer_0/Conv2D/ReadVariableOp2V
)model/conv_layer_1/BiasAdd/ReadVariableOp)model/conv_layer_1/BiasAdd/ReadVariableOp2T
(model/conv_layer_1/Conv2D/ReadVariableOp(model/conv_layer_1/Conv2D/ReadVariableOp2V
)model/conv_layer_2/BiasAdd/ReadVariableOp)model/conv_layer_2/BiasAdd/ReadVariableOp2T
(model/conv_layer_2/Conv2D/ReadVariableOp(model/conv_layer_2/Conv2D/ReadVariableOp2V
)model/conv_layer_3/BiasAdd/ReadVariableOp)model/conv_layer_3/BiasAdd/ReadVariableOp2T
(model/conv_layer_3/Conv2D/ReadVariableOp(model/conv_layer_3/Conv2D/ReadVariableOp2V
)model/conv_layer_4/BiasAdd/ReadVariableOp)model/conv_layer_4/BiasAdd/ReadVariableOp2T
(model/conv_layer_4/Conv2D/ReadVariableOp(model/conv_layer_4/Conv2D/ReadVariableOp2X
*model/dense_layer_0/BiasAdd/ReadVariableOp*model/dense_layer_0/BiasAdd/ReadVariableOp2V
)model/dense_layer_0/MatMul/ReadVariableOp)model/dense_layer_0/MatMul/ReadVariableOp2X
*model/dense_layer_1/BiasAdd/ReadVariableOp*model/dense_layer_1/BiasAdd/ReadVariableOp2V
)model/dense_layer_1/MatMul/ReadVariableOp)model/dense_layer_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_2_97866Y
>conv_layer_2_kernel_regularizer_l2loss_readvariableop_resource:@�
identity��5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp>conv_layer_2_kernel_regularizer_l2loss_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: e
IdentityIdentity'conv_layer_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp
�
h
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_95712

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������#�c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������#�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������#�:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
�
,__inference_conv_layer_2_layer_call_fn_97673

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_95701x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������#�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������G@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������G@
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_95626	
inputZ
Cperturbative_model_gaussian_model_tensordot_readvariableop_resource:_�G
=perturbative_model_gaussian_model_mul_readvariableop_resource: ^
Dperturbative_model_model_conv_layer_0_conv2d_readvariableop_resource: S
Eperturbative_model_model_conv_layer_0_biasadd_readvariableop_resource: ^
Dperturbative_model_model_conv_layer_1_conv2d_readvariableop_resource: @S
Eperturbative_model_model_conv_layer_1_biasadd_readvariableop_resource:@_
Dperturbative_model_model_conv_layer_2_conv2d_readvariableop_resource:@�T
Eperturbative_model_model_conv_layer_2_biasadd_readvariableop_resource:	�`
Dperturbative_model_model_conv_layer_3_conv2d_readvariableop_resource:��T
Eperturbative_model_model_conv_layer_3_biasadd_readvariableop_resource:	�`
Dperturbative_model_model_conv_layer_4_conv2d_readvariableop_resource:��T
Eperturbative_model_model_conv_layer_4_biasadd_readvariableop_resource:	�Y
Eperturbative_model_model_dense_layer_0_matmul_readvariableop_resource:
��U
Fperturbative_model_model_dense_layer_0_biasadd_readvariableop_resource:	�X
Eperturbative_model_model_dense_layer_1_matmul_readvariableop_resource:	�T
Fperturbative_model_model_dense_layer_1_biasadd_readvariableop_resource:
identity��:perturbative_model/gaussian_model/Tensordot/ReadVariableOp�4perturbative_model/gaussian_model/mul/ReadVariableOp�<perturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOp�;perturbative_model/model/conv_layer_0/Conv2D/ReadVariableOp�<perturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOp�;perturbative_model/model/conv_layer_1/Conv2D/ReadVariableOp�<perturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOp�;perturbative_model/model/conv_layer_2/Conv2D/ReadVariableOp�<perturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOp�;perturbative_model/model/conv_layer_3/Conv2D/ReadVariableOp�<perturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOp�;perturbative_model/model/conv_layer_4/Conv2D/ReadVariableOp�=perturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOp�<perturbative_model/model/dense_layer_0/MatMul/ReadVariableOp�=perturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOp�<perturbative_model/model/dense_layer_1/MatMul/ReadVariableOp�
:perturbative_model/gaussian_model/Tensordot/ReadVariableOpReadVariableOpCperturbative_model_gaussian_model_tensordot_readvariableop_resource*#
_output_shapes
:_�*
dtype0�
0perturbative_model/gaussian_model/Tensordot/axesConst*
_output_shapes
:*
dtype0*!
valueB"         z
0perturbative_model/gaussian_model/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: f
1perturbative_model/gaussian_model/Tensordot/ShapeShapeinput*
T0*
_output_shapes
:{
9perturbative_model/gaussian_model/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
4perturbative_model/gaussian_model/Tensordot/GatherV2GatherV2:perturbative_model/gaussian_model/Tensordot/Shape:output:09perturbative_model/gaussian_model/Tensordot/free:output:0Bperturbative_model/gaussian_model/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:}
;perturbative_model/gaussian_model/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
6perturbative_model/gaussian_model/Tensordot/GatherV2_1GatherV2:perturbative_model/gaussian_model/Tensordot/Shape:output:09perturbative_model/gaussian_model/Tensordot/axes:output:0Dperturbative_model/gaussian_model/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:{
1perturbative_model/gaussian_model/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
0perturbative_model/gaussian_model/Tensordot/ProdProd=perturbative_model/gaussian_model/Tensordot/GatherV2:output:0:perturbative_model/gaussian_model/Tensordot/Const:output:0*
T0*
_output_shapes
: }
3perturbative_model/gaussian_model/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
2perturbative_model/gaussian_model/Tensordot/Prod_1Prod?perturbative_model/gaussian_model/Tensordot/GatherV2_1:output:0<perturbative_model/gaussian_model/Tensordot/Const_1:output:0*
T0*
_output_shapes
: y
7perturbative_model/gaussian_model/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
2perturbative_model/gaussian_model/Tensordot/concatConcatV29perturbative_model/gaussian_model/Tensordot/free:output:09perturbative_model/gaussian_model/Tensordot/axes:output:0@perturbative_model/gaussian_model/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
1perturbative_model/gaussian_model/Tensordot/stackPack9perturbative_model/gaussian_model/Tensordot/Prod:output:0;perturbative_model/gaussian_model/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
5perturbative_model/gaussian_model/Tensordot/transpose	Transposeinput;perturbative_model/gaussian_model/Tensordot/concat:output:0*
T0*0
_output_shapes
:���������_��
3perturbative_model/gaussian_model/Tensordot/ReshapeReshape9perturbative_model/gaussian_model/Tensordot/transpose:y:0:perturbative_model/gaussian_model/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
;perturbative_model/gaussian_model/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"��     �
5perturbative_model/gaussian_model/Tensordot/Reshape_1ReshapeBperturbative_model/gaussian_model/Tensordot/ReadVariableOp:value:0Dperturbative_model/gaussian_model/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
2perturbative_model/gaussian_model/Tensordot/MatMulMatMul<perturbative_model/gaussian_model/Tensordot/Reshape:output:0>perturbative_model/gaussian_model/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������v
3perturbative_model/gaussian_model/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB {
9perturbative_model/gaussian_model/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
4perturbative_model/gaussian_model/Tensordot/concat_1ConcatV2=perturbative_model/gaussian_model/Tensordot/GatherV2:output:0<perturbative_model/gaussian_model/Tensordot/Const_2:output:0Bperturbative_model/gaussian_model/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+perturbative_model/gaussian_model/TensordotReshape<perturbative_model/gaussian_model/Tensordot/MatMul:product:0=perturbative_model/gaussian_model/Tensordot/concat_1:output:0*
T0*#
_output_shapes
:����������
1perturbative_model/gaussian_model/ones_like/ShapeShape4perturbative_model/gaussian_model/Tensordot:output:0*
T0*
_output_shapes
:v
1perturbative_model/gaussian_model/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
+perturbative_model/gaussian_model/ones_likeFill:perturbative_model/gaussian_model/ones_like/Shape:output:0:perturbative_model/gaussian_model/ones_like/Const:output:0*
T0*#
_output_shapes
:����������
4perturbative_model/gaussian_model/mul/ReadVariableOpReadVariableOp=perturbative_model_gaussian_model_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
%perturbative_model/gaussian_model/mulMul4perturbative_model/gaussian_model/ones_like:output:0<perturbative_model/gaussian_model/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:����������
'perturbative_model/gaussian_model/stackPack4perturbative_model/gaussian_model/Tensordot:output:0)perturbative_model/gaussian_model/mul:z:0*
N*
T0*'
_output_shapes
:���������*
axis����������
;perturbative_model/model/conv_layer_0/Conv2D/ReadVariableOpReadVariableOpDperturbative_model_model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
,perturbative_model/model/conv_layer_0/Conv2DConv2DinputCperturbative_model/model/conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
<perturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOpReadVariableOpEperturbative_model_model_conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
-perturbative_model/model/conv_layer_0/BiasAddBiasAdd5perturbative_model/model/conv_layer_0/Conv2D:output:0Dperturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
/perturbative_model/model/conv_activation_0/ReluRelu6perturbative_model/model/conv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
;perturbative_model/model/conv_layer_1/Conv2D/ReadVariableOpReadVariableOpDperturbative_model_model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
,perturbative_model/model/conv_layer_1/Conv2DConv2D=perturbative_model/model/conv_activation_0/Relu:activations:0Cperturbative_model/model/conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
<perturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOpReadVariableOpEperturbative_model_model_conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
-perturbative_model/model/conv_layer_1/BiasAddBiasAdd5perturbative_model/model/conv_layer_1/Conv2D:output:0Dperturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
/perturbative_model/model/conv_activation_1/ReluRelu6perturbative_model/model/conv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
;perturbative_model/model/conv_layer_2/Conv2D/ReadVariableOpReadVariableOpDperturbative_model_model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
,perturbative_model/model/conv_layer_2/Conv2DConv2D=perturbative_model/model/conv_activation_1/Relu:activations:0Cperturbative_model/model/conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
<perturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOpReadVariableOpEperturbative_model_model_conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-perturbative_model/model/conv_layer_2/BiasAddBiasAdd5perturbative_model/model/conv_layer_2/Conv2D:output:0Dperturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
/perturbative_model/model/conv_activation_2/ReluRelu6perturbative_model/model/conv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
;perturbative_model/model/conv_layer_3/Conv2D/ReadVariableOpReadVariableOpDperturbative_model_model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
,perturbative_model/model/conv_layer_3/Conv2DConv2D=perturbative_model/model/conv_activation_2/Relu:activations:0Cperturbative_model/model/conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
<perturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOpReadVariableOpEperturbative_model_model_conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-perturbative_model/model/conv_layer_3/BiasAddBiasAdd5perturbative_model/model/conv_layer_3/Conv2D:output:0Dperturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
/perturbative_model/model/conv_activation_3/ReluRelu6perturbative_model/model/conv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
;perturbative_model/model/conv_layer_4/Conv2D/ReadVariableOpReadVariableOpDperturbative_model_model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
,perturbative_model/model/conv_layer_4/Conv2DConv2D=perturbative_model/model/conv_activation_3/Relu:activations:0Cperturbative_model/model/conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
<perturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOpReadVariableOpEperturbative_model_model_conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-perturbative_model/model/conv_layer_4/BiasAddBiasAdd5perturbative_model/model/conv_layer_4/Conv2D:output:0Dperturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
/perturbative_model/model/conv_activation_4/ReluRelu6perturbative_model/model/conv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������w
&perturbative_model/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
(perturbative_model/model/flatten/ReshapeReshape=perturbative_model/model/conv_activation_4/Relu:activations:0/perturbative_model/model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
<perturbative_model/model/dense_layer_0/MatMul/ReadVariableOpReadVariableOpEperturbative_model_model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
-perturbative_model/model/dense_layer_0/MatMulMatMul1perturbative_model/model/flatten/Reshape:output:0Dperturbative_model/model/dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
=perturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOpReadVariableOpFperturbative_model_model_dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
.perturbative_model/model/dense_layer_0/BiasAddBiasAdd7perturbative_model/model/dense_layer_0/MatMul:product:0Eperturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
0perturbative_model/model/dense_activation_0/ReluRelu7perturbative_model/model/dense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
<perturbative_model/model/dense_layer_1/MatMul/ReadVariableOpReadVariableOpEperturbative_model_model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
-perturbative_model/model/dense_layer_1/MatMulMatMul>perturbative_model/model/dense_activation_0/Relu:activations:0Dperturbative_model/model/dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
=perturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOpReadVariableOpFperturbative_model_model_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
.perturbative_model/model/dense_layer_1/BiasAddBiasAdd7perturbative_model/model/dense_layer_1/MatMul:product:0Eperturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
perturbative_model/add/addAddV20perturbative_model/gaussian_model/stack:output:07perturbative_model/model/dense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
7perturbative_model/Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    �����
9perturbative_model/Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
9perturbative_model/Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
1perturbative_model/Sigma_Activation/strided_sliceStridedSliceperturbative_model/add/add:z:0@perturbative_model/Sigma_Activation/strided_slice/stack:output:0Bperturbative_model/Sigma_Activation/strided_slice/stack_1:output:0Bperturbative_model/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
,perturbative_model/Sigma_Activation/SoftplusSoftplus:perturbative_model/Sigma_Activation/strided_slice:output:0*
T0*'
_output_shapes
:����������
9perturbative_model/Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
;perturbative_model/Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    �����
;perturbative_model/Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
3perturbative_model/Sigma_Activation/strided_slice_1StridedSliceperturbative_model/add/add:z:0Bperturbative_model/Sigma_Activation/strided_slice_1/stack:output:0Dperturbative_model/Sigma_Activation/strided_slice_1/stack_1:output:0Dperturbative_model/Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskz
/perturbative_model/Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
*perturbative_model/Sigma_Activation/concatConcatV2<perturbative_model/Sigma_Activation/strided_slice_1:output:0:perturbative_model/Sigma_Activation/Softplus:activations:08perturbative_model/Sigma_Activation/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
IdentityIdentity3perturbative_model/Sigma_Activation/concat:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp;^perturbative_model/gaussian_model/Tensordot/ReadVariableOp5^perturbative_model/gaussian_model/mul/ReadVariableOp=^perturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOp<^perturbative_model/model/conv_layer_0/Conv2D/ReadVariableOp=^perturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOp<^perturbative_model/model/conv_layer_1/Conv2D/ReadVariableOp=^perturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOp<^perturbative_model/model/conv_layer_2/Conv2D/ReadVariableOp=^perturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOp<^perturbative_model/model/conv_layer_3/Conv2D/ReadVariableOp=^perturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOp<^perturbative_model/model/conv_layer_4/Conv2D/ReadVariableOp>^perturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOp=^perturbative_model/model/dense_layer_0/MatMul/ReadVariableOp>^perturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOp=^perturbative_model/model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2x
:perturbative_model/gaussian_model/Tensordot/ReadVariableOp:perturbative_model/gaussian_model/Tensordot/ReadVariableOp2l
4perturbative_model/gaussian_model/mul/ReadVariableOp4perturbative_model/gaussian_model/mul/ReadVariableOp2|
<perturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOp<perturbative_model/model/conv_layer_0/BiasAdd/ReadVariableOp2z
;perturbative_model/model/conv_layer_0/Conv2D/ReadVariableOp;perturbative_model/model/conv_layer_0/Conv2D/ReadVariableOp2|
<perturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOp<perturbative_model/model/conv_layer_1/BiasAdd/ReadVariableOp2z
;perturbative_model/model/conv_layer_1/Conv2D/ReadVariableOp;perturbative_model/model/conv_layer_1/Conv2D/ReadVariableOp2|
<perturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOp<perturbative_model/model/conv_layer_2/BiasAdd/ReadVariableOp2z
;perturbative_model/model/conv_layer_2/Conv2D/ReadVariableOp;perturbative_model/model/conv_layer_2/Conv2D/ReadVariableOp2|
<perturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOp<perturbative_model/model/conv_layer_3/BiasAdd/ReadVariableOp2z
;perturbative_model/model/conv_layer_3/Conv2D/ReadVariableOp;perturbative_model/model/conv_layer_3/Conv2D/ReadVariableOp2|
<perturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOp<perturbative_model/model/conv_layer_4/BiasAdd/ReadVariableOp2z
;perturbative_model/model/conv_layer_4/Conv2D/ReadVariableOp;perturbative_model/model/conv_layer_4/Conv2D/ReadVariableOp2~
=perturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOp=perturbative_model/model/dense_layer_0/BiasAdd/ReadVariableOp2|
<perturbative_model/model/dense_layer_0/MatMul/ReadVariableOp<perturbative_model/model/dense_layer_0/MatMul/ReadVariableOp2~
=perturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOp=perturbative_model/model/dense_layer_1/BiasAdd/ReadVariableOp2|
<perturbative_model/model/dense_layer_1/MatMul/ReadVariableOp<perturbative_model/model/dense_layer_1/MatMul/ReadVariableOp:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�C
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96463

inputs+
gaussian_model_96375:_�
gaussian_model_96377: %
model_96380: 
model_96382: %
model_96384: @
model_96386:@&
model_96388:@�
model_96390:	�'
model_96392:��
model_96394:	�'
model_96396:��
model_96398:	�
model_96400:
��
model_96402:	�
model_96404:	�
model_96406:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&gaussian_model/StatefulPartitionedCall�model/StatefulPartitionedCall�
&gaussian_model/StatefulPartitionedCallStatefulPartitionedCallinputsgaussian_model_96375gaussian_model_96377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_96380model_96382model_96384model_96386model_96388model_96390model_96392model_96394model_96396model_96398model_96400model_96402model_96404model_96406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_95858�
add/PartitionedCallPartitionedCall/gaussian_model/StatefulPartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_96415�
 Sigma_Activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96380*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96384*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96388*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96392*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96396*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96400* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96404*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)Sigma_Activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^gaussian_model/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&gaussian_model/StatefulPartitionedCall&gaussian_model/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_97631

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:���������/� c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:���������/� "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������/� :X T
0
_output_shapes
:���������/� 
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_97621

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������/� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
i
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:����������[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�$
!__inference__traced_restore_98277
file_prefix;
$assignvariableop_gaussian_model_proj:_�1
'assignvariableop_1_gaussian_model_sigma: @
&assignvariableop_2_conv_layer_0_kernel: 2
$assignvariableop_3_conv_layer_0_bias: @
&assignvariableop_4_conv_layer_1_kernel: @2
$assignvariableop_5_conv_layer_1_bias:@A
&assignvariableop_6_conv_layer_2_kernel:@�3
$assignvariableop_7_conv_layer_2_bias:	�B
&assignvariableop_8_conv_layer_3_kernel:��3
$assignvariableop_9_conv_layer_3_bias:	�C
'assignvariableop_10_conv_layer_4_kernel:��4
%assignvariableop_11_conv_layer_4_bias:	�<
(assignvariableop_12_dense_layer_0_kernel:
��5
&assignvariableop_13_dense_layer_0_bias:	�;
(assignvariableop_14_dense_layer_1_kernel:	�4
&assignvariableop_15_dense_layer_1_bias:$
assignvariableop_16_beta_1: $
assignvariableop_17_beta_2: #
assignvariableop_18_decay: +
!assignvariableop_19_learning_rate: '
assignvariableop_20_adam_iter:	 %
assignvariableop_21_total_3: %
assignvariableop_22_count_3: %
assignvariableop_23_total_2: %
assignvariableop_24_count_2: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: H
.assignvariableop_29_adam_conv_layer_0_kernel_m: :
,assignvariableop_30_adam_conv_layer_0_bias_m: H
.assignvariableop_31_adam_conv_layer_1_kernel_m: @:
,assignvariableop_32_adam_conv_layer_1_bias_m:@I
.assignvariableop_33_adam_conv_layer_2_kernel_m:@�;
,assignvariableop_34_adam_conv_layer_2_bias_m:	�J
.assignvariableop_35_adam_conv_layer_3_kernel_m:��;
,assignvariableop_36_adam_conv_layer_3_bias_m:	�J
.assignvariableop_37_adam_conv_layer_4_kernel_m:��;
,assignvariableop_38_adam_conv_layer_4_bias_m:	�C
/assignvariableop_39_adam_dense_layer_0_kernel_m:
��<
-assignvariableop_40_adam_dense_layer_0_bias_m:	�B
/assignvariableop_41_adam_dense_layer_1_kernel_m:	�;
-assignvariableop_42_adam_dense_layer_1_bias_m:H
.assignvariableop_43_adam_conv_layer_0_kernel_v: :
,assignvariableop_44_adam_conv_layer_0_bias_v: H
.assignvariableop_45_adam_conv_layer_1_kernel_v: @:
,assignvariableop_46_adam_conv_layer_1_bias_v:@I
.assignvariableop_47_adam_conv_layer_2_kernel_v:@�;
,assignvariableop_48_adam_conv_layer_2_bias_v:	�J
.assignvariableop_49_adam_conv_layer_3_kernel_v:��;
,assignvariableop_50_adam_conv_layer_3_bias_v:	�J
.assignvariableop_51_adam_conv_layer_4_kernel_v:��;
,assignvariableop_52_adam_conv_layer_4_bias_v:	�C
/assignvariableop_53_adam_dense_layer_0_kernel_v:
��<
-assignvariableop_54_adam_dense_layer_0_bias_v:	�B
/assignvariableop_55_adam_dense_layer_1_kernel_v:	�;
-assignvariableop_56_adam_dense_layer_1_bias_v:
identity_58��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B4layer_with_weights-0/proj/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp$assignvariableop_gaussian_model_projIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp'assignvariableop_1_gaussian_model_sigmaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv_layer_0_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv_layer_0_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv_layer_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv_layer_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_conv_layer_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_conv_layer_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_conv_layer_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_conv_layer_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_conv_layer_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_conv_layer_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_dense_layer_0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_dense_layer_0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_dense_layer_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp&assignvariableop_15_dense_layer_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_3Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_3Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_conv_layer_0_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_conv_layer_0_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_conv_layer_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_conv_layer_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_conv_layer_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv_layer_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv_layer_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv_layer_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_conv_layer_4_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_conv_layer_4_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp/assignvariableop_39_adam_dense_layer_0_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp-assignvariableop_40_adam_dense_layer_0_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp/assignvariableop_41_adam_dense_layer_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp-assignvariableop_42_adam_dense_layer_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_conv_layer_0_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_conv_layer_0_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_conv_layer_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_conv_layer_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_conv_layer_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_conv_layer_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp.assignvariableop_49_adam_conv_layer_3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_conv_layer_3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_conv_layer_4_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_conv_layer_4_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp/assignvariableop_53_adam_dense_layer_0_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp-assignvariableop_54_adam_dense_layer_0_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp/assignvariableop_55_adam_dense_layer_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp-assignvariableop_56_adam_dense_layer_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*�
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
M
1__inference_conv_activation_3_layer_call_fn_97725

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_95739i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97137

inputsG
0gaussian_model_tensordot_readvariableop_resource:_�4
*gaussian_model_mul_readvariableop_resource: K
1model_conv_layer_0_conv2d_readvariableop_resource: @
2model_conv_layer_0_biasadd_readvariableop_resource: K
1model_conv_layer_1_conv2d_readvariableop_resource: @@
2model_conv_layer_1_biasadd_readvariableop_resource:@L
1model_conv_layer_2_conv2d_readvariableop_resource:@�A
2model_conv_layer_2_biasadd_readvariableop_resource:	�M
1model_conv_layer_3_conv2d_readvariableop_resource:��A
2model_conv_layer_3_biasadd_readvariableop_resource:	�M
1model_conv_layer_4_conv2d_readvariableop_resource:��A
2model_conv_layer_4_biasadd_readvariableop_resource:	�F
2model_dense_layer_0_matmul_readvariableop_resource:
��B
3model_dense_layer_0_biasadd_readvariableop_resource:	�E
2model_dense_layer_1_matmul_readvariableop_resource:	�A
3model_dense_layer_1_biasadd_readvariableop_resource:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�'gaussian_model/Tensordot/ReadVariableOp�!gaussian_model/mul/ReadVariableOp�)model/conv_layer_0/BiasAdd/ReadVariableOp�(model/conv_layer_0/Conv2D/ReadVariableOp�)model/conv_layer_1/BiasAdd/ReadVariableOp�(model/conv_layer_1/Conv2D/ReadVariableOp�)model/conv_layer_2/BiasAdd/ReadVariableOp�(model/conv_layer_2/Conv2D/ReadVariableOp�)model/conv_layer_3/BiasAdd/ReadVariableOp�(model/conv_layer_3/Conv2D/ReadVariableOp�)model/conv_layer_4/BiasAdd/ReadVariableOp�(model/conv_layer_4/Conv2D/ReadVariableOp�*model/dense_layer_0/BiasAdd/ReadVariableOp�)model/dense_layer_0/MatMul/ReadVariableOp�*model/dense_layer_1/BiasAdd/ReadVariableOp�)model/dense_layer_1/MatMul/ReadVariableOp�
'gaussian_model/Tensordot/ReadVariableOpReadVariableOp0gaussian_model_tensordot_readvariableop_resource*#
_output_shapes
:_�*
dtype0r
gaussian_model/Tensordot/axesConst*
_output_shapes
:*
dtype0*!
valueB"         g
gaussian_model/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB: T
gaussian_model/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:h
&gaussian_model/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!gaussian_model/Tensordot/GatherV2GatherV2'gaussian_model/Tensordot/Shape:output:0&gaussian_model/Tensordot/free:output:0/gaussian_model/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(gaussian_model/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
#gaussian_model/Tensordot/GatherV2_1GatherV2'gaussian_model/Tensordot/Shape:output:0&gaussian_model/Tensordot/axes:output:01gaussian_model/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
gaussian_model/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gaussian_model/Tensordot/ProdProd*gaussian_model/Tensordot/GatherV2:output:0'gaussian_model/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 gaussian_model/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gaussian_model/Tensordot/Prod_1Prod,gaussian_model/Tensordot/GatherV2_1:output:0)gaussian_model/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$gaussian_model/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
gaussian_model/Tensordot/concatConcatV2&gaussian_model/Tensordot/free:output:0&gaussian_model/Tensordot/axes:output:0-gaussian_model/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
gaussian_model/Tensordot/stackPack&gaussian_model/Tensordot/Prod:output:0(gaussian_model/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
"gaussian_model/Tensordot/transpose	Transposeinputs(gaussian_model/Tensordot/concat:output:0*
T0*0
_output_shapes
:���������_��
 gaussian_model/Tensordot/ReshapeReshape&gaussian_model/Tensordot/transpose:y:0'gaussian_model/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������y
(gaussian_model/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"��     �
"gaussian_model/Tensordot/Reshape_1Reshape/gaussian_model/Tensordot/ReadVariableOp:value:01gaussian_model/Tensordot/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
���
gaussian_model/Tensordot/MatMulMatMul)gaussian_model/Tensordot/Reshape:output:0+gaussian_model/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:���������c
 gaussian_model/Tensordot/Const_2Const*
_output_shapes
: *
dtype0*
valueB h
&gaussian_model/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!gaussian_model/Tensordot/concat_1ConcatV2*gaussian_model/Tensordot/GatherV2:output:0)gaussian_model/Tensordot/Const_2:output:0/gaussian_model/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
gaussian_model/TensordotReshape)gaussian_model/Tensordot/MatMul:product:0*gaussian_model/Tensordot/concat_1:output:0*
T0*#
_output_shapes
:���������o
gaussian_model/ones_like/ShapeShape!gaussian_model/Tensordot:output:0*
T0*
_output_shapes
:c
gaussian_model/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
gaussian_model/ones_likeFill'gaussian_model/ones_like/Shape:output:0'gaussian_model/ones_like/Const:output:0*
T0*#
_output_shapes
:����������
!gaussian_model/mul/ReadVariableOpReadVariableOp*gaussian_model_mul_readvariableop_resource*
_output_shapes
: *
dtype0�
gaussian_model/mulMul!gaussian_model/ones_like:output:0)gaussian_model/mul/ReadVariableOp:value:0*
T0*#
_output_shapes
:����������
gaussian_model/stackPack!gaussian_model/Tensordot:output:0gaussian_model/mul:z:0*
N*
T0*'
_output_shapes
:���������*
axis����������
(model/conv_layer_0/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
model/conv_layer_0/Conv2DConv2Dinputs0model/conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
)model/conv_layer_0/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
model/conv_layer_0/BiasAddBiasAdd"model/conv_layer_0/Conv2D:output:01model/conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
model/conv_activation_0/ReluRelu#model/conv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
(model/conv_layer_1/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
model/conv_layer_1/Conv2DConv2D*model/conv_activation_0/Relu:activations:00model/conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
)model/conv_layer_1/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/conv_layer_1/BiasAddBiasAdd"model/conv_layer_1/Conv2D:output:01model/conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
model/conv_activation_1/ReluRelu#model/conv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
(model/conv_layer_2/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
model/conv_layer_2/Conv2DConv2D*model/conv_activation_1/Relu:activations:00model/conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
)model/conv_layer_2/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_2/BiasAddBiasAdd"model/conv_layer_2/Conv2D:output:01model/conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
model/conv_activation_2/ReluRelu#model/conv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
(model/conv_layer_3/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv_layer_3/Conv2DConv2D*model/conv_activation_2/Relu:activations:00model/conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
)model/conv_layer_3/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_3/BiasAddBiasAdd"model/conv_layer_3/Conv2D:output:01model/conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
model/conv_activation_3/ReluRelu#model/conv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
(model/conv_layer_4/Conv2D/ReadVariableOpReadVariableOp1model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
model/conv_layer_4/Conv2DConv2D*model/conv_activation_3/Relu:activations:00model/conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
)model/conv_layer_4/BiasAdd/ReadVariableOpReadVariableOp2model_conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/conv_layer_4/BiasAddBiasAdd"model/conv_layer_4/Conv2D:output:01model/conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
model/conv_activation_4/ReluRelu#model/conv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
model/flatten/ReshapeReshape*model/conv_activation_4/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
)model/dense_layer_0/MatMul/ReadVariableOpReadVariableOp2model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense_layer_0/MatMulMatMulmodel/flatten/Reshape:output:01model/dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*model/dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp3model_dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense_layer_0/BiasAddBiasAdd$model/dense_layer_0/MatMul:product:02model/dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������~
model/dense_activation_0/ReluRelu$model/dense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
)model/dense_layer_1/MatMul/ReadVariableOpReadVariableOp2model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/dense_layer_1/MatMulMatMul+model/dense_activation_0/Relu:activations:01model/dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*model/dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp3model_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/dense_layer_1/BiasAddBiasAdd$model/dense_layer_1/MatMul:product:02model/dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
add/addAddV2gaussian_model/stack:output:0$model/dense_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������u
$Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����w
&Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        w
&Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Sigma_Activation/strided_sliceStridedSliceadd/add:z:0-Sigma_Activation/strided_slice/stack:output:0/Sigma_Activation/strided_slice/stack_1:output:0/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
Sigma_Activation/SoftplusSoftplus'Sigma_Activation/strided_slice:output:0*
T0*'
_output_shapes
:���������w
&Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����y
(Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 Sigma_Activation/strided_slice_1StridedSliceadd/add:z:0/Sigma_Activation/strided_slice_1/stack:output:01Sigma_Activation/strided_slice_1/stack_1:output:01Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskg
Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Sigma_Activation/concatConcatV2)Sigma_Activation/strided_slice_1:output:0'Sigma_Activation/Softplus:activations:0%Sigma_Activation/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp1model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp2model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp2model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: o
IdentityIdentity Sigma_Activation/concat:output:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp(^gaussian_model/Tensordot/ReadVariableOp"^gaussian_model/mul/ReadVariableOp*^model/conv_layer_0/BiasAdd/ReadVariableOp)^model/conv_layer_0/Conv2D/ReadVariableOp*^model/conv_layer_1/BiasAdd/ReadVariableOp)^model/conv_layer_1/Conv2D/ReadVariableOp*^model/conv_layer_2/BiasAdd/ReadVariableOp)^model/conv_layer_2/Conv2D/ReadVariableOp*^model/conv_layer_3/BiasAdd/ReadVariableOp)^model/conv_layer_3/Conv2D/ReadVariableOp*^model/conv_layer_4/BiasAdd/ReadVariableOp)^model/conv_layer_4/Conv2D/ReadVariableOp+^model/dense_layer_0/BiasAdd/ReadVariableOp*^model/dense_layer_0/MatMul/ReadVariableOp+^model/dense_layer_1/BiasAdd/ReadVariableOp*^model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2R
'gaussian_model/Tensordot/ReadVariableOp'gaussian_model/Tensordot/ReadVariableOp2F
!gaussian_model/mul/ReadVariableOp!gaussian_model/mul/ReadVariableOp2V
)model/conv_layer_0/BiasAdd/ReadVariableOp)model/conv_layer_0/BiasAdd/ReadVariableOp2T
(model/conv_layer_0/Conv2D/ReadVariableOp(model/conv_layer_0/Conv2D/ReadVariableOp2V
)model/conv_layer_1/BiasAdd/ReadVariableOp)model/conv_layer_1/BiasAdd/ReadVariableOp2T
(model/conv_layer_1/Conv2D/ReadVariableOp(model/conv_layer_1/Conv2D/ReadVariableOp2V
)model/conv_layer_2/BiasAdd/ReadVariableOp)model/conv_layer_2/BiasAdd/ReadVariableOp2T
(model/conv_layer_2/Conv2D/ReadVariableOp(model/conv_layer_2/Conv2D/ReadVariableOp2V
)model/conv_layer_3/BiasAdd/ReadVariableOp)model/conv_layer_3/BiasAdd/ReadVariableOp2T
(model/conv_layer_3/Conv2D/ReadVariableOp(model/conv_layer_3/Conv2D/ReadVariableOp2V
)model/conv_layer_4/BiasAdd/ReadVariableOp)model/conv_layer_4/BiasAdd/ReadVariableOp2T
(model/conv_layer_4/Conv2D/ReadVariableOp(model/conv_layer_4/Conv2D/ReadVariableOp2X
*model/dense_layer_0/BiasAdd/ReadVariableOp*model/dense_layer_0/BiasAdd/ReadVariableOp2V
)model/dense_layer_0/MatMul/ReadVariableOp)model/dense_layer_0/MatMul/ReadVariableOp2X
*model/dense_layer_1/BiasAdd/ReadVariableOp*model/dense_layer_1/BiasAdd/ReadVariableOp2V
)model/dense_layer_1/MatMul/ReadVariableOp)model/dense_layer_1/MatMul/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�m
�
__inference__traced_save_98096
file_prefix2
.savev2_gaussian_model_proj_read_readvariableop3
/savev2_gaussian_model_sigma_read_readvariableop2
.savev2_conv_layer_0_kernel_read_readvariableop0
,savev2_conv_layer_0_bias_read_readvariableop2
.savev2_conv_layer_1_kernel_read_readvariableop0
,savev2_conv_layer_1_bias_read_readvariableop2
.savev2_conv_layer_2_kernel_read_readvariableop0
,savev2_conv_layer_2_bias_read_readvariableop2
.savev2_conv_layer_3_kernel_read_readvariableop0
,savev2_conv_layer_3_bias_read_readvariableop2
.savev2_conv_layer_4_kernel_read_readvariableop0
,savev2_conv_layer_4_bias_read_readvariableop3
/savev2_dense_layer_0_kernel_read_readvariableop1
-savev2_dense_layer_0_bias_read_readvariableop3
/savev2_dense_layer_1_kernel_read_readvariableop1
-savev2_dense_layer_1_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_conv_layer_0_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_0_bias_m_read_readvariableop9
5savev2_adam_conv_layer_1_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_1_bias_m_read_readvariableop9
5savev2_adam_conv_layer_2_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_2_bias_m_read_readvariableop9
5savev2_adam_conv_layer_3_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_3_bias_m_read_readvariableop9
5savev2_adam_conv_layer_4_kernel_m_read_readvariableop7
3savev2_adam_conv_layer_4_bias_m_read_readvariableop:
6savev2_adam_dense_layer_0_kernel_m_read_readvariableop8
4savev2_adam_dense_layer_0_bias_m_read_readvariableop:
6savev2_adam_dense_layer_1_kernel_m_read_readvariableop8
4savev2_adam_dense_layer_1_bias_m_read_readvariableop9
5savev2_adam_conv_layer_0_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_0_bias_v_read_readvariableop9
5savev2_adam_conv_layer_1_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_1_bias_v_read_readvariableop9
5savev2_adam_conv_layer_2_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_2_bias_v_read_readvariableop9
5savev2_adam_conv_layer_3_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_3_bias_v_read_readvariableop9
5savev2_adam_conv_layer_4_kernel_v_read_readvariableop7
3savev2_adam_conv_layer_4_bias_v_read_readvariableop:
6savev2_adam_dense_layer_0_kernel_v_read_readvariableop8
4savev2_adam_dense_layer_0_bias_v_read_readvariableop:
6savev2_adam_dense_layer_1_kernel_v_read_readvariableop8
4savev2_adam_dense_layer_1_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value�B�:B4layer_with_weights-0/proj/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/sigma/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*�
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_gaussian_model_proj_read_readvariableop/savev2_gaussian_model_sigma_read_readvariableop.savev2_conv_layer_0_kernel_read_readvariableop,savev2_conv_layer_0_bias_read_readvariableop.savev2_conv_layer_1_kernel_read_readvariableop,savev2_conv_layer_1_bias_read_readvariableop.savev2_conv_layer_2_kernel_read_readvariableop,savev2_conv_layer_2_bias_read_readvariableop.savev2_conv_layer_3_kernel_read_readvariableop,savev2_conv_layer_3_bias_read_readvariableop.savev2_conv_layer_4_kernel_read_readvariableop,savev2_conv_layer_4_bias_read_readvariableop/savev2_dense_layer_0_kernel_read_readvariableop-savev2_dense_layer_0_bias_read_readvariableop/savev2_dense_layer_1_kernel_read_readvariableop-savev2_dense_layer_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_conv_layer_0_kernel_m_read_readvariableop3savev2_adam_conv_layer_0_bias_m_read_readvariableop5savev2_adam_conv_layer_1_kernel_m_read_readvariableop3savev2_adam_conv_layer_1_bias_m_read_readvariableop5savev2_adam_conv_layer_2_kernel_m_read_readvariableop3savev2_adam_conv_layer_2_bias_m_read_readvariableop5savev2_adam_conv_layer_3_kernel_m_read_readvariableop3savev2_adam_conv_layer_3_bias_m_read_readvariableop5savev2_adam_conv_layer_4_kernel_m_read_readvariableop3savev2_adam_conv_layer_4_bias_m_read_readvariableop6savev2_adam_dense_layer_0_kernel_m_read_readvariableop4savev2_adam_dense_layer_0_bias_m_read_readvariableop6savev2_adam_dense_layer_1_kernel_m_read_readvariableop4savev2_adam_dense_layer_1_bias_m_read_readvariableop5savev2_adam_conv_layer_0_kernel_v_read_readvariableop3savev2_adam_conv_layer_0_bias_v_read_readvariableop5savev2_adam_conv_layer_1_kernel_v_read_readvariableop3savev2_adam_conv_layer_1_bias_v_read_readvariableop5savev2_adam_conv_layer_2_kernel_v_read_readvariableop3savev2_adam_conv_layer_2_bias_v_read_readvariableop5savev2_adam_conv_layer_3_kernel_v_read_readvariableop3savev2_adam_conv_layer_3_bias_v_read_readvariableop5savev2_adam_conv_layer_4_kernel_v_read_readvariableop3savev2_adam_conv_layer_4_bias_v_read_readvariableop6savev2_adam_dense_layer_0_kernel_v_read_readvariableop4savev2_adam_dense_layer_0_bias_v_read_readvariableop6savev2_adam_dense_layer_1_kernel_v_read_readvariableop4savev2_adam_dense_layer_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :_�: : : : @:@:@�:�:��:�:��:�:
��:�:	�:: : : : : : : : : : : : : : : : @:@:@�:�:��:�:��:�:
��:�:	�:: : : @:@:@�:�:��:�:��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:_�:

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:-")
'
_output_shapes
:@�:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:.&*
(
_output_shapes
:��:!'

_output_shapes	
:�:&("
 
_output_shapes
:
��:!)

_output_shapes	
:�:%*!

_output_shapes
:	�: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@:-0)
'
_output_shapes
:@�:!1

_output_shapes	
:�:.2*
(
_output_shapes
:��:!3

_output_shapes	
:�:.4*
(
_output_shapes
:��:!5

_output_shapes	
:�:&6"
 
_output_shapes
:
��:!7

_output_shapes	
:�:%8!

_output_shapes
:	�: 9

_output_shapes
:::

_output_shapes
: 
�
�
%__inference_model_layer_call_fn_96181	
input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_96117o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�C
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96836	
input+
gaussian_model_96771:_�
gaussian_model_96773: %
model_96776: 
model_96778: %
model_96780: @
model_96782:@&
model_96784:@�
model_96786:	�'
model_96788:��
model_96790:	�'
model_96792:��
model_96794:	�
model_96796:
��
model_96798:	�
model_96800:	�
model_96802:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&gaussian_model/StatefulPartitionedCall�model/StatefulPartitionedCall�
&gaussian_model/StatefulPartitionedCallStatefulPartitionedCallinputgaussian_model_96771gaussian_model_96773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374�
model/StatefulPartitionedCallStatefulPartitionedCallinputmodel_96776model_96778model_96780model_96782model_96784model_96786model_96788model_96790model_96792model_96794model_96796model_96798model_96800model_96802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_96117�
add/PartitionedCallPartitionedCall/gaussian_model/StatefulPartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_96415�
 Sigma_Activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96776*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96780*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96784*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96788*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96792*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96796* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96800*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)Sigma_Activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^gaussian_model/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&gaussian_model/StatefulPartitionedCall&gaussian_model/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�
�
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_97720

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������#�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������#�
 
_user_specified_nameinputs
�
h
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_97730

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:����������c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_96909	
input
unknown:_�
	unknown_0: #
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
��

unknown_12:	�

unknown_13:	�

unknown_14:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__wrapped_model_95626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
0
_output_shapes
:���������_�

_user_specified_nameinput
�
N
2__inference_dense_activation_0_layer_call_fn_97802

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_95801a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
N
2__inference_dense_activation_1_layer_call_fn_97835

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_95647

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:���������/� �
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������_�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_95827

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_97598
x
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask^
SoftplusSoftplusstrided_slice:output:0*
T0*'
_output_shapes
:���������f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskV
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2strided_slice_1:output:0Softplus:activations:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:J F
'
_output_shapes
:���������

_user_specified_namex
�
�
-__inference_dense_layer_0_layer_call_fn_97783

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_95790p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�C
�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96628

inputs+
gaussian_model_96563:_�
gaussian_model_96565: %
model_96568: 
model_96570: %
model_96572: @
model_96574:@&
model_96576:@�
model_96578:	�'
model_96580:��
model_96582:	�'
model_96584:��
model_96586:	�
model_96588:
��
model_96590:	�
model_96592:	�
model_96594:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�&gaussian_model/StatefulPartitionedCall�model/StatefulPartitionedCall�
&gaussian_model/StatefulPartitionedCallStatefulPartitionedCallinputsgaussian_model_96563gaussian_model_96565*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *R
fMRK
I__inference_gaussian_model_layer_call_and_return_conditional_losses_96374�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_96568model_96570model_96572model_96574model_96576model_96578model_96580model_96582model_96584model_96586model_96588model_96590model_96592model_96594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_96117�
add/PartitionedCallPartitionedCall/gaussian_model/StatefulPartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_96415�
 Sigma_Activation/PartitionedCallPartitionedCalladd/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_96432�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96568*&
_output_shapes
: *
dtype0�
&conv_layer_0/kernel/Regularizer/L2LossL2Loss=conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_0/kernel/Regularizer/mulMul.conv_layer_0/kernel/Regularizer/mul/x:output:0/conv_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96572*&
_output_shapes
: @*
dtype0�
&conv_layer_1/kernel/Regularizer/L2LossL2Loss=conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_1/kernel/Regularizer/mulMul.conv_layer_1/kernel/Regularizer/mul/x:output:0/conv_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96576*'
_output_shapes
:@�*
dtype0�
&conv_layer_2/kernel/Regularizer/L2LossL2Loss=conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_2/kernel/Regularizer/mulMul.conv_layer_2/kernel/Regularizer/mul/x:output:0/conv_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96580*(
_output_shapes
:��*
dtype0�
&conv_layer_3/kernel/Regularizer/L2LossL2Loss=conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_3/kernel/Regularizer/mulMul.conv_layer_3/kernel/Regularizer/mul/x:output:0/conv_layer_3/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96584*(
_output_shapes
:��*
dtype0�
&conv_layer_4/kernel/Regularizer/L2LossL2Loss=conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: j
%conv_layer_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�ŧ7�
#conv_layer_4/kernel/Regularizer/mulMul.conv_layer_4/kernel/Regularizer/mul/x:output:0/conv_layer_4/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96588* 
_output_shapes
:
��*
dtype0�
'dense_layer_0/kernel/Regularizer/L2LossL2Loss>dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_0/kernel/Regularizer/mulMul/dense_layer_0/kernel/Regularizer/mul/x:output:00dense_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: �
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_96592*
_output_shapes
:	�*
dtype0�
'dense_layer_1/kernel/Regularizer/L2LossL2Loss>dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: k
&dense_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o�<�
$dense_layer_1/kernel/Regularizer/mulMul/dense_layer_1/kernel/Regularizer/mul/x:output:00dense_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)Sigma_Activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp'^gaussian_model/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:���������_�: : : : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2P
&gaussian_model/StatefulPartitionedCall&gaussian_model/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
@
input7
serving_default_input:0���������_�D
Sigma_Activation0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
proj
	sigma"
_tf_keras_layer
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
 layer_with_weights-4
 layer-9
!layer-10
"layer-11
#layer_with_weights-5
#layer-12
$layer-13
%layer_with_weights-6
%layer-14
&layer-15
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_network
�
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0
1
92
:3
;4
<5
=6
>7
?8
@9
A10
B11
C12
D13
E14
F15"
trackable_list_wrapper
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32�
2__inference_perturbative_model_layer_call_fn_96498
2__inference_perturbative_model_layer_call_fn_96974
2__inference_perturbative_model_layer_call_fn_97011
2__inference_perturbative_model_layer_call_fn_96700�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
�
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97137
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97263
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96768
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96836�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
�B�
 __inference__wrapped_model_95626input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�

Tbeta_1

Ubeta_2
	Vdecay
Wlearning_rate
Xiter9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�Cm�Dm�Em�Fm�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�Cv�Dv�Ev�Fv�"
	optimizer
,
Yserving_default"
signature_map
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
_trace_02�
.__inference_gaussian_model_layer_call_fn_97272�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z_trace_0
�
`trace_02�
I__inference_gaussian_model_layer_call_and_return_conditional_losses_97308�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z`trace_0
(:&_�2gaussian_model/proj
: 2gaussian_model/sigma
"
_tf_keras_input_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

9kernel
:bias
 g_jit_compiled_convolution_op"
_tf_keras_layer
�
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
�
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

;kernel
<bias
 t_jit_compiled_convolution_op"
_tf_keras_layer
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses

=kernel
>bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

?kernel
@bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Akernel
Bbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13"
trackable_list_wrapper
�
90
:1
;2
<3
=4
>5
?6
@7
A8
B9
C10
D11
E12
F13"
trackable_list_wrapper
X
�0
�1
�2
�3
�4
�5
�6"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_95889
%__inference_model_layer_call_fn_97369
%__inference_model_layer_call_fn_97402
%__inference_model_layer_call_fn_96181�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
@__inference_model_layer_call_and_return_conditional_losses_97484
@__inference_model_layer_call_and_return_conditional_losses_97566
@__inference_model_layer_call_and_return_conditional_losses_96256
@__inference_model_layer_call_and_return_conditional_losses_96331�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
#__inference_add_layer_call_fn_97572�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
>__inference_add_layer_call_and_return_conditional_losses_97578�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_Sigma_Activation_layer_call_fn_97583�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_97598�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
-:+ 2conv_layer_0/kernel
: 2conv_layer_0/bias
-:+ @2conv_layer_1/kernel
:@2conv_layer_1/bias
.:,@�2conv_layer_2/kernel
 :�2conv_layer_2/bias
/:-��2conv_layer_3/kernel
 :�2conv_layer_3/bias
/:-��2conv_layer_4/kernel
 :�2conv_layer_4/bias
(:&
��2dense_layer_0/kernel
!:�2dense_layer_0/bias
':%	�2dense_layer_1/kernel
 :2dense_layer_1/bias
.
0
1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_perturbative_model_layer_call_fn_96498input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_perturbative_model_layer_call_fn_96974inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_perturbative_model_layer_call_fn_97011inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_perturbative_model_layer_call_fn_96700input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97137inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97263inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96768input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96836input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
�B�
#__inference_signature_wrapper_96909input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_gaussian_model_layer_call_fn_97272x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_gaussian_model_layer_call_and_return_conditional_losses_97308x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_0_layer_call_fn_97607�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_97621�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_0_layer_call_fn_97626�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_97631�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_1_layer_call_fn_97640�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_97654�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_1_layer_call_fn_97659�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_97664�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_2_layer_call_fn_97673�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_97687�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_2_layer_call_fn_97692�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_97697�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_3_layer_call_fn_97706�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_97720�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_3_layer_call_fn_97725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_97730�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_4_layer_call_fn_97739�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_97753�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_4_layer_call_fn_97758�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_97763�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_97768�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_97774�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_layer_0_layer_call_fn_97783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_97797�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_dense_activation_0_layer_call_fn_97802�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_97807�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_layer_1_layer_call_fn_97816�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_97830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_dense_activation_1_layer_call_fn_97835�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_97839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
__inference_loss_fn_0_97848�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_1_97857�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_2_97866�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_3_97875�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_4_97884�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_5_97893�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
�
�trace_02�
__inference_loss_fn_6_97902�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
 9
!10
"11
#12
$13
%14
&15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_95889input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_97369inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_97402inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_model_layer_call_fn_96181input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_97484inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_97566inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_96256input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_model_layer_call_and_return_conditional_losses_96331input"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_add_layer_call_fn_97572inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
>__inference_add_layer_call_and_return_conditional_losses_97578inputs/0inputs/1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_Sigma_Activation_layer_call_fn_97583x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_97598x"�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_layer_0_layer_call_fn_97607inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_97621inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_activation_0_layer_call_fn_97626inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_97631inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_layer_1_layer_call_fn_97640inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_97654inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_activation_1_layer_call_fn_97659inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_97664inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_layer_2_layer_call_fn_97673inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_97687inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_activation_2_layer_call_fn_97692inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_97697inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_layer_3_layer_call_fn_97706inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_97720inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_activation_3_layer_call_fn_97725inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_97730inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_conv_layer_4_layer_call_fn_97739inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_97753inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_conv_activation_4_layer_call_fn_97758inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_97763inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_flatten_layer_call_fn_97768inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_flatten_layer_call_and_return_conditional_losses_97774inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_layer_0_layer_call_fn_97783inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_97797inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_dense_activation_0_layer_call_fn_97802inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_97807inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dense_layer_1_layer_call_fn_97816inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_97830inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_dense_activation_1_layer_call_fn_97835inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_97839inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_97848"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_1_97857"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_2_97866"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_3_97875"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_4_97884"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_5_97893"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
__inference_loss_fn_6_97902"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
2:0 2Adam/conv_layer_0/kernel/m
$:" 2Adam/conv_layer_0/bias/m
2:0 @2Adam/conv_layer_1/kernel/m
$:"@2Adam/conv_layer_1/bias/m
3:1@�2Adam/conv_layer_2/kernel/m
%:#�2Adam/conv_layer_2/bias/m
4:2��2Adam/conv_layer_3/kernel/m
%:#�2Adam/conv_layer_3/bias/m
4:2��2Adam/conv_layer_4/kernel/m
%:#�2Adam/conv_layer_4/bias/m
-:+
��2Adam/dense_layer_0/kernel/m
&:$�2Adam/dense_layer_0/bias/m
,:*	�2Adam/dense_layer_1/kernel/m
%:#2Adam/dense_layer_1/bias/m
2:0 2Adam/conv_layer_0/kernel/v
$:" 2Adam/conv_layer_0/bias/v
2:0 @2Adam/conv_layer_1/kernel/v
$:"@2Adam/conv_layer_1/bias/v
3:1@�2Adam/conv_layer_2/kernel/v
%:#�2Adam/conv_layer_2/bias/v
4:2��2Adam/conv_layer_3/kernel/v
%:#�2Adam/conv_layer_3/bias/v
4:2��2Adam/conv_layer_4/kernel/v
%:#�2Adam/conv_layer_4/bias/v
-:+
��2Adam/dense_layer_0/kernel/v
&:$�2Adam/dense_layer_0/bias/v
,:*	�2Adam/dense_layer_1/kernel/v
%:#2Adam/dense_layer_1/bias/v�
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_97598S*�'
 �
�
x���������
� "%�"
�
0���������
� z
0__inference_Sigma_Activation_layer_call_fn_97583F*�'
 �
�
x���������
� "�����������
 __inference__wrapped_model_95626�9:;<=>?@ABCDEF7�4
-�*
(�%
input���������_�
� "C�@
>
Sigma_Activation*�'
Sigma_Activation����������
>__inference_add_layer_call_and_return_conditional_losses_97578�Z�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
#__inference_add_layer_call_fn_97572vZ�W
P�M
K�H
"�
inputs/0���������
"�
inputs/1���������
� "�����������
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_97631j8�5
.�+
)�&
inputs���������/� 
� ".�+
$�!
0���������/� 
� �
1__inference_conv_activation_0_layer_call_fn_97626]8�5
.�+
)�&
inputs���������/� 
� "!����������/� �
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_97664h7�4
-�*
(�%
inputs���������G@
� "-�*
#� 
0���������G@
� �
1__inference_conv_activation_1_layer_call_fn_97659[7�4
-�*
(�%
inputs���������G@
� " ����������G@�
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_97697j8�5
.�+
)�&
inputs���������#�
� ".�+
$�!
0���������#�
� �
1__inference_conv_activation_2_layer_call_fn_97692]8�5
.�+
)�&
inputs���������#�
� "!����������#��
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_97730j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
1__inference_conv_activation_3_layer_call_fn_97725]8�5
.�+
)�&
inputs����������
� "!������������
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_97763j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
1__inference_conv_activation_4_layer_call_fn_97758]8�5
.�+
)�&
inputs����������
� "!������������
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_97621n9:8�5
.�+
)�&
inputs���������_�
� ".�+
$�!
0���������/� 
� �
,__inference_conv_layer_0_layer_call_fn_97607a9:8�5
.�+
)�&
inputs���������_�
� "!����������/� �
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_97654m;<8�5
.�+
)�&
inputs���������/� 
� "-�*
#� 
0���������G@
� �
,__inference_conv_layer_1_layer_call_fn_97640`;<8�5
.�+
)�&
inputs���������/� 
� " ����������G@�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_97687m=>7�4
-�*
(�%
inputs���������G@
� ".�+
$�!
0���������#�
� �
,__inference_conv_layer_2_layer_call_fn_97673`=>7�4
-�*
(�%
inputs���������G@
� "!����������#��
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_97720n?@8�5
.�+
)�&
inputs���������#�
� ".�+
$�!
0����������
� �
,__inference_conv_layer_3_layer_call_fn_97706a?@8�5
.�+
)�&
inputs���������#�
� "!������������
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_97753nAB8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_conv_layer_4_layer_call_fn_97739aAB8�5
.�+
)�&
inputs����������
� "!������������
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_97807Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_dense_activation_0_layer_call_fn_97802M0�-
&�#
!�
inputs����������
� "������������
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_97839X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
2__inference_dense_activation_1_layer_call_fn_97835K/�,
%�"
 �
inputs���������
� "�����������
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_97797^CD0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
-__inference_dense_layer_0_layer_call_fn_97783QCD0�-
&�#
!�
inputs����������
� "������������
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_97830]EF0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
-__inference_dense_layer_1_layer_call_fn_97816PEF0�-
&�#
!�
inputs����������
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_97774b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
'__inference_flatten_layer_call_fn_97768U8�5
.�+
)�&
inputs����������
� "������������
I__inference_gaussian_model_layer_call_and_return_conditional_losses_97308`3�0
)�&
$�!
x���������_�
� "%�"
�
0���������
� �
.__inference_gaussian_model_layer_call_fn_97272S3�0
)�&
$�!
x���������_�
� "����������:
__inference_loss_fn_0_978489�

� 
� "� :
__inference_loss_fn_1_97857;�

� 
� "� :
__inference_loss_fn_2_97866=�

� 
� "� :
__inference_loss_fn_3_97875?�

� 
� "� :
__inference_loss_fn_4_97884A�

� 
� "� :
__inference_loss_fn_5_97893C�

� 
� "� :
__inference_loss_fn_6_97902E�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_96256x9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_96331x9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_97484y9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p 

 
� "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_97566y9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p

 
� "%�"
�
0���������
� �
%__inference_model_layer_call_fn_95889k9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p 

 
� "�����������
%__inference_model_layer_call_fn_96181k9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p

 
� "�����������
%__inference_model_layer_call_fn_97369l9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p 

 
� "�����������
%__inference_model_layer_call_fn_97402l9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p

 
� "�����������
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96768z9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p 

 
� "%�"
�
0���������
� �
M__inference_perturbative_model_layer_call_and_return_conditional_losses_96836z9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p

 
� "%�"
�
0���������
� �
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97137{9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p 

 
� "%�"
�
0���������
� �
M__inference_perturbative_model_layer_call_and_return_conditional_losses_97263{9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p

 
� "%�"
�
0���������
� �
2__inference_perturbative_model_layer_call_fn_96498m9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p 

 
� "�����������
2__inference_perturbative_model_layer_call_fn_96700m9:;<=>?@ABCDEF?�<
5�2
(�%
input���������_�
p

 
� "�����������
2__inference_perturbative_model_layer_call_fn_96974n9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p 

 
� "�����������
2__inference_perturbative_model_layer_call_fn_97011n9:;<=>?@ABCDEF@�=
6�3
)�&
inputs���������_�
p

 
� "�����������
#__inference_signature_wrapper_96909�9:;<=>?@ABCDEF@�=
� 
6�3
1
input(�%
input���������_�"C�@
>
Sigma_Activation*�'
Sigma_Activation���������