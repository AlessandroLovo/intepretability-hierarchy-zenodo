��
��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
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
�
serving_default_model_inputPlaceholder*0
_output_shapes
:���������_�*
dtype0*%
shape:���������_�
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_model_inputconv_layer_0/kernelconv_layer_0/biasconv_layer_1/kernelconv_layer_1/biasconv_layer_2/kernelconv_layer_2/biasconv_layer_3/kernelconv_layer_3/biasconv_layer_4/kernelconv_layer_4/biasdense_layer_0/kerneldense_layer_0/biasdense_layer_1/kerneldense_layer_1/bias*
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *,
f'R%
#__inference_signature_wrapper_66013

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ߋ
valueԋBЋ Bȋ
�
layer_with_weights-0
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses* 
j
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513*
j
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513*
* 
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
;trace_0
<trace_1
=trace_2
>trace_3* 
6
?trace_0
@trace_1
Atrace_2
Btrace_3* 
* 
�

Cbeta_1

Dbeta_2
	Edecay
Flearning_rate
Giter(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�*

Hserving_default* 
* 
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

(kernel
)bias
 O_jit_compiled_convolution_op*
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

*kernel
+bias
 \_jit_compiled_convolution_op*
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses* 
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

,kernel
-bias
 i_jit_compiled_convolution_op*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

.kernel
/bias
 v_jit_compiled_convolution_op*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses* 
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

0kernel
1bias
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

2kernel
3bias*
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

4kernel
5bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
j
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513*
j
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513*
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
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
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
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
SM
VARIABLE_VALUEconv_layer_0/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_0/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEconv_layer_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEconv_layer_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_layer_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_layer_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEdense_layer_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdense_layer_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*
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
(0
)1*

(0
)1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
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
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

*0
+1*

*0
+1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
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
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

,0
-1*

,0
-1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
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
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

.0
/1*

.0
/1*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
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
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

00
11*

00
11*


�0* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
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
20
31*

20
31*


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
40
51*

40
51*


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
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
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
VARIABLE_VALUEAdam/conv_layer_0/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_0/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_0/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_0/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/conv_layer_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEAdam/conv_layer_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/dense_layer_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUEAdam/dense_layer_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'conv_layer_0/kernel/Read/ReadVariableOp%conv_layer_0/bias/Read/ReadVariableOp'conv_layer_1/kernel/Read/ReadVariableOp%conv_layer_1/bias/Read/ReadVariableOp'conv_layer_2/kernel/Read/ReadVariableOp%conv_layer_2/bias/Read/ReadVariableOp'conv_layer_3/kernel/Read/ReadVariableOp%conv_layer_3/bias/Read/ReadVariableOp'conv_layer_4/kernel/Read/ReadVariableOp%conv_layer_4/bias/Read/ReadVariableOp(dense_layer_0/kernel/Read/ReadVariableOp&dense_layer_0/bias/Read/ReadVariableOp(dense_layer_1/kernel/Read/ReadVariableOp&dense_layer_1/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/conv_layer_0/kernel/m/Read/ReadVariableOp,Adam/conv_layer_0/bias/m/Read/ReadVariableOp.Adam/conv_layer_1/kernel/m/Read/ReadVariableOp,Adam/conv_layer_1/bias/m/Read/ReadVariableOp.Adam/conv_layer_2/kernel/m/Read/ReadVariableOp,Adam/conv_layer_2/bias/m/Read/ReadVariableOp.Adam/conv_layer_3/kernel/m/Read/ReadVariableOp,Adam/conv_layer_3/bias/m/Read/ReadVariableOp.Adam/conv_layer_4/kernel/m/Read/ReadVariableOp,Adam/conv_layer_4/bias/m/Read/ReadVariableOp/Adam/dense_layer_0/kernel/m/Read/ReadVariableOp-Adam/dense_layer_0/bias/m/Read/ReadVariableOp/Adam/dense_layer_1/kernel/m/Read/ReadVariableOp-Adam/dense_layer_1/bias/m/Read/ReadVariableOp.Adam/conv_layer_0/kernel/v/Read/ReadVariableOp,Adam/conv_layer_0/bias/v/Read/ReadVariableOp.Adam/conv_layer_1/kernel/v/Read/ReadVariableOp,Adam/conv_layer_1/bias/v/Read/ReadVariableOp.Adam/conv_layer_2/kernel/v/Read/ReadVariableOp,Adam/conv_layer_2/bias/v/Read/ReadVariableOp.Adam/conv_layer_3/kernel/v/Read/ReadVariableOp,Adam/conv_layer_3/bias/v/Read/ReadVariableOp.Adam/conv_layer_4/kernel/v/Read/ReadVariableOp,Adam/conv_layer_4/bias/v/Read/ReadVariableOp/Adam/dense_layer_0/kernel/v/Read/ReadVariableOp-Adam/dense_layer_0/bias/v/Read/ReadVariableOp/Adam/dense_layer_1/kernel/v/Read/ReadVariableOp-Adam/dense_layer_1/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *'
f"R 
__inference__traced_save_67063
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_layer_0/kernelconv_layer_0/biasconv_layer_1/kernelconv_layer_1/biasconv_layer_2/kernelconv_layer_2/biasconv_layer_3/kernelconv_layer_3/biasconv_layer_4/kernelconv_layer_4/biasdense_layer_0/kerneldense_layer_0/biasdense_layer_1/kerneldense_layer_1/biasbeta_1beta_2decaylearning_rate	Adam/itertotal_3count_3total_2count_2total_1count_1totalcountAdam/conv_layer_0/kernel/mAdam/conv_layer_0/bias/mAdam/conv_layer_1/kernel/mAdam/conv_layer_1/bias/mAdam/conv_layer_2/kernel/mAdam/conv_layer_2/bias/mAdam/conv_layer_3/kernel/mAdam/conv_layer_3/bias/mAdam/conv_layer_4/kernel/mAdam/conv_layer_4/bias/mAdam/dense_layer_0/kernel/mAdam/dense_layer_0/bias/mAdam/dense_layer_1/kernel/mAdam/dense_layer_1/bias/mAdam/conv_layer_0/kernel/vAdam/conv_layer_0/bias/vAdam/conv_layer_1/kernel/vAdam/conv_layer_1/bias/vAdam/conv_layer_2/kernel/vAdam/conv_layer_2/bias/vAdam/conv_layer_3/kernel/vAdam/conv_layer_3/bias/vAdam/conv_layer_4/kernel/vAdam/conv_layer_4/bias/vAdam/dense_layer_0/kernel/vAdam/dense_layer_0/bias/vAdam/dense_layer_1/kernel/vAdam/dense_layer_1/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� **
f%R#
!__inference__traced_restore_67238��
�
�
,__inference_conv_layer_3_layer_call_fn_66679

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937x
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
�
�
,__inference_conv_layer_4_layer_call_fn_66712

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964x
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
�
M
1__inference_conv_activation_2_layer_call_fn_66665

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921i
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
�
�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_66627

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
�
b
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591
x
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask`
SoftplusSoftplusstrided_slice_1:output:0*
T0*'
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2strided_slice:output:0Softplus:activations:0concat/axis:output:0*
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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_64983

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
�
�
,__inference_conv_layer_1_layer_call_fn_66613

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883w
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
�
�
%__inference_model_layer_call_fn_65390	
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65326o
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
�
h
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_66670

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
�j
�
__inference__traced_save_67063
file_prefix2
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_conv_layer_0_kernel_read_readvariableop,savev2_conv_layer_0_bias_read_readvariableop.savev2_conv_layer_1_kernel_read_readvariableop,savev2_conv_layer_1_bias_read_readvariableop.savev2_conv_layer_2_kernel_read_readvariableop,savev2_conv_layer_2_bias_read_readvariableop.savev2_conv_layer_3_kernel_read_readvariableop,savev2_conv_layer_3_bias_read_readvariableop.savev2_conv_layer_4_kernel_read_readvariableop,savev2_conv_layer_4_bias_read_readvariableop/savev2_dense_layer_0_kernel_read_readvariableop-savev2_dense_layer_0_bias_read_readvariableop/savev2_dense_layer_1_kernel_read_readvariableop-savev2_dense_layer_1_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_conv_layer_0_kernel_m_read_readvariableop3savev2_adam_conv_layer_0_bias_m_read_readvariableop5savev2_adam_conv_layer_1_kernel_m_read_readvariableop3savev2_adam_conv_layer_1_bias_m_read_readvariableop5savev2_adam_conv_layer_2_kernel_m_read_readvariableop3savev2_adam_conv_layer_2_bias_m_read_readvariableop5savev2_adam_conv_layer_3_kernel_m_read_readvariableop3savev2_adam_conv_layer_3_bias_m_read_readvariableop5savev2_adam_conv_layer_4_kernel_m_read_readvariableop3savev2_adam_conv_layer_4_bias_m_read_readvariableop6savev2_adam_dense_layer_0_kernel_m_read_readvariableop4savev2_adam_dense_layer_0_bias_m_read_readvariableop6savev2_adam_dense_layer_1_kernel_m_read_readvariableop4savev2_adam_dense_layer_1_bias_m_read_readvariableop5savev2_adam_conv_layer_0_kernel_v_read_readvariableop3savev2_adam_conv_layer_0_bias_v_read_readvariableop5savev2_adam_conv_layer_1_kernel_v_read_readvariableop3savev2_adam_conv_layer_1_bias_v_read_readvariableop5savev2_adam_conv_layer_2_kernel_v_read_readvariableop3savev2_adam_conv_layer_2_bias_v_read_readvariableop5savev2_adam_conv_layer_3_kernel_v_read_readvariableop3savev2_adam_conv_layer_3_bias_v_read_readvariableop5savev2_adam_conv_layer_4_kernel_v_read_readvariableop3savev2_adam_conv_layer_4_bias_v_read_readvariableop6savev2_adam_dense_layer_0_kernel_v_read_readvariableop4savev2_adam_dense_layer_0_bias_v_read_readvariableop6savev2_adam_dense_layer_1_kernel_v_read_readvariableop4savev2_adam_dense_layer_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	�
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
�: : : : @:@:@�:�:��:�:��:�:
��:�:	�:: : : : : : : : : : : : : : : : @:@:@�:�:��:�:��:�:
��:�:	�:: : : @:@:@�:�:��:�:��:�:
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:- )
'
_output_shapes
:@�:!!

_output_shapes	
:�:."*
(
_output_shapes
:��:!#

_output_shapes	
:�:.$*
(
_output_shapes
:��:!%

_output_shapes	
:�:&&"
 
_output_shapes
:
��:!'

_output_shapes	
:�:%(!

_output_shapes
:	�: )

_output_shapes
::,*(
&
_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@:-.)
'
_output_shapes
:@�:!/

_output_shapes	
:�:.0*
(
_output_shapes
:��:!1

_output_shapes	
:�:.2*
(
_output_shapes
:��:!3

_output_shapes	
:�:&4"
 
_output_shapes
:
��:!5

_output_shapes	
:�:%6!

_output_shapes
:	�: 7

_output_shapes
::8

_output_shapes
: 
�
�
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856

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
�l
�

@__inference_model_layer_call_and_return_conditional_losses_65540	
input,
conv_layer_0_65468:  
conv_layer_0_65470: ,
conv_layer_1_65474: @ 
conv_layer_1_65476:@-
conv_layer_2_65480:@�!
conv_layer_2_65482:	�.
conv_layer_3_65486:��!
conv_layer_3_65488:	�.
conv_layer_4_65492:��!
conv_layer_4_65494:	�'
dense_layer_0_65499:
��"
dense_layer_0_65501:	�&
dense_layer_1_65505:	�!
dense_layer_1_65507:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputconv_layer_0_65468conv_layer_0_65470*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_65474conv_layer_1_65476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_65480conv_layer_2_65482*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_65486conv_layer_3_65488*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_65492conv_layer_4_65494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_64983�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_65499dense_layer_0_65501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_65505dense_layer_1_65507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_65468*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_65474*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_65480*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_65486*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_65492*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_65499* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_65505*
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
�
*__inference_sequential_layer_call_fn_65820
model_input!
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65756o
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
StatefulPartitionedCallStatefulPartitionedCall:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�
�
-__inference_dense_layer_1_layer_call_fn_66789

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026o
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
�
�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_66770

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
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_66594

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
�l
�

@__inference_model_layer_call_and_return_conditional_losses_65326

inputs,
conv_layer_0_65254:  
conv_layer_0_65256: ,
conv_layer_1_65260: @ 
conv_layer_1_65262:@-
conv_layer_2_65266:@�!
conv_layer_2_65268:	�.
conv_layer_3_65272:��!
conv_layer_3_65274:	�.
conv_layer_4_65278:��!
conv_layer_4_65280:	�'
dense_layer_0_65285:
��"
dense_layer_0_65287:	�&
dense_layer_1_65291:	�!
dense_layer_1_65293:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_0_65254conv_layer_0_65256*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_65260conv_layer_1_65262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_65266conv_layer_2_65268*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_65272conv_layer_3_65274*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_65278conv_layer_4_65280*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_64983�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_65285dense_layer_0_65287*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_65291dense_layer_1_65293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_65254*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_65260*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_65266*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_65272*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_65278*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_65285* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_65291*
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
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_65622

inputs%
model_65547: 
model_65549: %
model_65551: @
model_65553:@&
model_65555:@�
model_65557:	�'
model_65559:��
model_65561:	�'
model_65563:��
model_65565:	�
model_65567:
��
model_65569:	�
model_65571:	�
model_65573:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�model/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_65547model_65549model_65551model_65553model_65555model_65557model_65559model_65561model_65563model_65565model_65567model_65569model_65571model_65573*
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65067�
 Sigma_Activation/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65547*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65551*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65555*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65559*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65563*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65567* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65571*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�
�
%__inference_model_layer_call_fn_66354

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65067o
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
�
C
'__inference_flatten_layer_call_fn_66741

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_64983a
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
�
M
1__inference_conv_activation_0_layer_call_fn_66599

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867i
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
�l
�
 __inference__wrapped_model_64835
model_inputV
<sequential_model_conv_layer_0_conv2d_readvariableop_resource: K
=sequential_model_conv_layer_0_biasadd_readvariableop_resource: V
<sequential_model_conv_layer_1_conv2d_readvariableop_resource: @K
=sequential_model_conv_layer_1_biasadd_readvariableop_resource:@W
<sequential_model_conv_layer_2_conv2d_readvariableop_resource:@�L
=sequential_model_conv_layer_2_biasadd_readvariableop_resource:	�X
<sequential_model_conv_layer_3_conv2d_readvariableop_resource:��L
=sequential_model_conv_layer_3_biasadd_readvariableop_resource:	�X
<sequential_model_conv_layer_4_conv2d_readvariableop_resource:��L
=sequential_model_conv_layer_4_biasadd_readvariableop_resource:	�Q
=sequential_model_dense_layer_0_matmul_readvariableop_resource:
��M
>sequential_model_dense_layer_0_biasadd_readvariableop_resource:	�P
=sequential_model_dense_layer_1_matmul_readvariableop_resource:	�L
>sequential_model_dense_layer_1_biasadd_readvariableop_resource:
identity��4sequential/model/conv_layer_0/BiasAdd/ReadVariableOp�3sequential/model/conv_layer_0/Conv2D/ReadVariableOp�4sequential/model/conv_layer_1/BiasAdd/ReadVariableOp�3sequential/model/conv_layer_1/Conv2D/ReadVariableOp�4sequential/model/conv_layer_2/BiasAdd/ReadVariableOp�3sequential/model/conv_layer_2/Conv2D/ReadVariableOp�4sequential/model/conv_layer_3/BiasAdd/ReadVariableOp�3sequential/model/conv_layer_3/Conv2D/ReadVariableOp�4sequential/model/conv_layer_4/BiasAdd/ReadVariableOp�3sequential/model/conv_layer_4/Conv2D/ReadVariableOp�5sequential/model/dense_layer_0/BiasAdd/ReadVariableOp�4sequential/model/dense_layer_0/MatMul/ReadVariableOp�5sequential/model/dense_layer_1/BiasAdd/ReadVariableOp�4sequential/model/dense_layer_1/MatMul/ReadVariableOp�
3sequential/model/conv_layer_0/Conv2D/ReadVariableOpReadVariableOp<sequential_model_conv_layer_0_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
$sequential/model/conv_layer_0/Conv2DConv2Dmodel_input;sequential/model/conv_layer_0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� *
paddingVALID*
strides
�
4sequential/model/conv_layer_0/BiasAdd/ReadVariableOpReadVariableOp=sequential_model_conv_layer_0_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
%sequential/model/conv_layer_0/BiasAddBiasAdd-sequential/model/conv_layer_0/Conv2D:output:0<sequential/model/conv_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������/� �
'sequential/model/conv_activation_0/ReluRelu.sequential/model/conv_layer_0/BiasAdd:output:0*
T0*0
_output_shapes
:���������/� �
3sequential/model/conv_layer_1/Conv2D/ReadVariableOpReadVariableOp<sequential_model_conv_layer_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
$sequential/model/conv_layer_1/Conv2DConv2D5sequential/model/conv_activation_0/Relu:activations:0;sequential/model/conv_layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@*
paddingVALID*
strides
�
4sequential/model/conv_layer_1/BiasAdd/ReadVariableOpReadVariableOp=sequential_model_conv_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%sequential/model/conv_layer_1/BiasAddBiasAdd-sequential/model/conv_layer_1/Conv2D:output:0<sequential/model/conv_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������G@�
'sequential/model/conv_activation_1/ReluRelu.sequential/model/conv_layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������G@�
3sequential/model/conv_layer_2/Conv2D/ReadVariableOpReadVariableOp<sequential_model_conv_layer_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
$sequential/model/conv_layer_2/Conv2DConv2D5sequential/model/conv_activation_1/Relu:activations:0;sequential/model/conv_layer_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#�*
paddingVALID*
strides
�
4sequential/model/conv_layer_2/BiasAdd/ReadVariableOpReadVariableOp=sequential_model_conv_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%sequential/model/conv_layer_2/BiasAddBiasAdd-sequential/model/conv_layer_2/Conv2D:output:0<sequential/model/conv_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������#��
'sequential/model/conv_activation_2/ReluRelu.sequential/model/conv_layer_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������#��
3sequential/model/conv_layer_3/Conv2D/ReadVariableOpReadVariableOp<sequential_model_conv_layer_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$sequential/model/conv_layer_3/Conv2DConv2D5sequential/model/conv_activation_2/Relu:activations:0;sequential/model/conv_layer_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
4sequential/model/conv_layer_3/BiasAdd/ReadVariableOpReadVariableOp=sequential_model_conv_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%sequential/model/conv_layer_3/BiasAddBiasAdd-sequential/model/conv_layer_3/Conv2D:output:0<sequential/model/conv_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
'sequential/model/conv_activation_3/ReluRelu.sequential/model/conv_layer_3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3sequential/model/conv_layer_4/Conv2D/ReadVariableOpReadVariableOp<sequential_model_conv_layer_4_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$sequential/model/conv_layer_4/Conv2DConv2D5sequential/model/conv_activation_3/Relu:activations:0;sequential/model/conv_layer_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
4sequential/model/conv_layer_4/BiasAdd/ReadVariableOpReadVariableOp=sequential_model_conv_layer_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%sequential/model/conv_layer_4/BiasAddBiasAdd-sequential/model/conv_layer_4/Conv2D:output:0<sequential/model/conv_layer_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
'sequential/model/conv_activation_4/ReluRelu.sequential/model/conv_layer_4/BiasAdd:output:0*
T0*0
_output_shapes
:����������o
sequential/model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
 sequential/model/flatten/ReshapeReshape5sequential/model/conv_activation_4/Relu:activations:0'sequential/model/flatten/Const:output:0*
T0*(
_output_shapes
:�����������
4sequential/model/dense_layer_0/MatMul/ReadVariableOpReadVariableOp=sequential_model_dense_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
%sequential/model/dense_layer_0/MatMulMatMul)sequential/model/flatten/Reshape:output:0<sequential/model/dense_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
5sequential/model/dense_layer_0/BiasAdd/ReadVariableOpReadVariableOp>sequential_model_dense_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&sequential/model/dense_layer_0/BiasAddBiasAdd/sequential/model/dense_layer_0/MatMul:product:0=sequential/model/dense_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(sequential/model/dense_activation_0/ReluRelu/sequential/model/dense_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
4sequential/model/dense_layer_1/MatMul/ReadVariableOpReadVariableOp=sequential_model_dense_layer_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%sequential/model/dense_layer_1/MatMulMatMul6sequential/model/dense_activation_0/Relu:activations:0<sequential/model/dense_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5sequential/model/dense_layer_1/BiasAdd/ReadVariableOpReadVariableOp>sequential_model_dense_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&sequential/model/dense_layer_1/BiasAddBiasAdd/sequential/model/dense_layer_1/MatMul:product:0=sequential/model/dense_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential/Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
1sequential/Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    �����
1sequential/Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
)sequential/Sigma_Activation/strided_sliceStridedSlice/sequential/model/dense_layer_1/BiasAdd:output:08sequential/Sigma_Activation/strided_slice/stack:output:0:sequential/Sigma_Activation/strided_slice/stack_1:output:0:sequential/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask�
1sequential/Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    �����
3sequential/Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        �
3sequential/Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
+sequential/Sigma_Activation/strided_slice_1StridedSlice/sequential/model/dense_layer_1/BiasAdd:output:0:sequential/Sigma_Activation/strided_slice_1/stack:output:0<sequential/Sigma_Activation/strided_slice_1/stack_1:output:0<sequential/Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
$sequential/Sigma_Activation/SoftplusSoftplus4sequential/Sigma_Activation/strided_slice_1:output:0*
T0*'
_output_shapes
:���������r
'sequential/Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
"sequential/Sigma_Activation/concatConcatV22sequential/Sigma_Activation/strided_slice:output:02sequential/Sigma_Activation/Softplus:activations:00sequential/Sigma_Activation/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������z
IdentityIdentity+sequential/Sigma_Activation/concat:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp5^sequential/model/conv_layer_0/BiasAdd/ReadVariableOp4^sequential/model/conv_layer_0/Conv2D/ReadVariableOp5^sequential/model/conv_layer_1/BiasAdd/ReadVariableOp4^sequential/model/conv_layer_1/Conv2D/ReadVariableOp5^sequential/model/conv_layer_2/BiasAdd/ReadVariableOp4^sequential/model/conv_layer_2/Conv2D/ReadVariableOp5^sequential/model/conv_layer_3/BiasAdd/ReadVariableOp4^sequential/model/conv_layer_3/Conv2D/ReadVariableOp5^sequential/model/conv_layer_4/BiasAdd/ReadVariableOp4^sequential/model/conv_layer_4/Conv2D/ReadVariableOp6^sequential/model/dense_layer_0/BiasAdd/ReadVariableOp5^sequential/model/dense_layer_0/MatMul/ReadVariableOp6^sequential/model/dense_layer_1/BiasAdd/ReadVariableOp5^sequential/model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2l
4sequential/model/conv_layer_0/BiasAdd/ReadVariableOp4sequential/model/conv_layer_0/BiasAdd/ReadVariableOp2j
3sequential/model/conv_layer_0/Conv2D/ReadVariableOp3sequential/model/conv_layer_0/Conv2D/ReadVariableOp2l
4sequential/model/conv_layer_1/BiasAdd/ReadVariableOp4sequential/model/conv_layer_1/BiasAdd/ReadVariableOp2j
3sequential/model/conv_layer_1/Conv2D/ReadVariableOp3sequential/model/conv_layer_1/Conv2D/ReadVariableOp2l
4sequential/model/conv_layer_2/BiasAdd/ReadVariableOp4sequential/model/conv_layer_2/BiasAdd/ReadVariableOp2j
3sequential/model/conv_layer_2/Conv2D/ReadVariableOp3sequential/model/conv_layer_2/Conv2D/ReadVariableOp2l
4sequential/model/conv_layer_3/BiasAdd/ReadVariableOp4sequential/model/conv_layer_3/BiasAdd/ReadVariableOp2j
3sequential/model/conv_layer_3/Conv2D/ReadVariableOp3sequential/model/conv_layer_3/Conv2D/ReadVariableOp2l
4sequential/model/conv_layer_4/BiasAdd/ReadVariableOp4sequential/model/conv_layer_4/BiasAdd/ReadVariableOp2j
3sequential/model/conv_layer_4/Conv2D/ReadVariableOp3sequential/model/conv_layer_4/Conv2D/ReadVariableOp2n
5sequential/model/dense_layer_0/BiasAdd/ReadVariableOp5sequential/model/dense_layer_0/BiasAdd/ReadVariableOp2l
4sequential/model/dense_layer_0/MatMul/ReadVariableOp4sequential/model/dense_layer_0/MatMul/ReadVariableOp2n
5sequential/model/dense_layer_1/BiasAdd/ReadVariableOp5sequential/model/dense_layer_1/BiasAdd/ReadVariableOp2l
4sequential/model/dense_layer_1/MatMul/ReadVariableOp4sequential/model/dense_layer_1/MatMul/ReadVariableOp:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�
b
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_66571
x
identityd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlicexstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask`
SoftplusSoftplusstrided_slice_1:output:0*
T0*'
_output_shapes
:���������V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
concatConcatV2strided_slice:output:0Softplus:activations:0concat/axis:output:0*
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
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66200

inputsK
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
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�)model/conv_layer_0/BiasAdd/ReadVariableOp�(model/conv_layer_0/Conv2D/ReadVariableOp�)model/conv_layer_1/BiasAdd/ReadVariableOp�(model/conv_layer_1/Conv2D/ReadVariableOp�)model/conv_layer_2/BiasAdd/ReadVariableOp�(model/conv_layer_2/Conv2D/ReadVariableOp�)model/conv_layer_3/BiasAdd/ReadVariableOp�(model/conv_layer_3/Conv2D/ReadVariableOp�)model/conv_layer_4/BiasAdd/ReadVariableOp�(model/conv_layer_4/Conv2D/ReadVariableOp�*model/dense_layer_0/BiasAdd/ReadVariableOp�)model/dense_layer_0/MatMul/ReadVariableOp�*model/dense_layer_1/BiasAdd/ReadVariableOp�)model/dense_layer_1/MatMul/ReadVariableOp�
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
:���������u
$Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����w
&Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Sigma_Activation/strided_sliceStridedSlice$model/dense_layer_1/BiasAdd:output:0-Sigma_Activation/strided_slice/stack:output:0/Sigma_Activation/strided_slice/stack_1:output:0/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskw
&Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����y
(Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 Sigma_Activation/strided_slice_1StridedSlice$model/dense_layer_1/BiasAdd:output:0/Sigma_Activation/strided_slice_1/stack:output:01Sigma_Activation/strided_slice_1/stack_1:output:01Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
Sigma_Activation/SoftplusSoftplus)Sigma_Activation/strided_slice_1:output:0*
T0*'
_output_shapes
:���������g
Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Sigma_Activation/concatConcatV2'Sigma_Activation/strided_slice:output:0'Sigma_Activation/Softplus:activations:0%Sigma_Activation/concat/axis:output:0*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*^model/conv_layer_0/BiasAdd/ReadVariableOp)^model/conv_layer_0/Conv2D/ReadVariableOp*^model/conv_layer_1/BiasAdd/ReadVariableOp)^model/conv_layer_1/Conv2D/ReadVariableOp*^model/conv_layer_2/BiasAdd/ReadVariableOp)^model/conv_layer_2/Conv2D/ReadVariableOp*^model/conv_layer_3/BiasAdd/ReadVariableOp)^model/conv_layer_3/Conv2D/ReadVariableOp*^model/conv_layer_4/BiasAdd/ReadVariableOp)^model/conv_layer_4/Conv2D/ReadVariableOp+^model/dense_layer_0/BiasAdd/ReadVariableOp*^model/dense_layer_0/MatMul/ReadVariableOp+^model/dense_layer_1/BiasAdd/ReadVariableOp*^model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2V
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
�
N
2__inference_dense_activation_0_layer_call_fn_66775

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010a
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
�
�
,__inference_conv_layer_0_layer_call_fn_66580

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856x
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
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_66660

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
�
i
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_66780

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
�
�
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999

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
�t
�
@__inference_model_layer_call_and_return_conditional_losses_66551

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
�
M
1__inference_conv_activation_1_layer_call_fn_66632

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894h
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
�	
�
__inference_loss_fn_3_66848Z
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
-__inference_dense_layer_0_layer_call_fn_66756

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999p
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
�
�
%__inference_model_layer_call_fn_65098	
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65067o
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
�
�
*__inference_sequential_layer_call_fn_66107

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
identity��StatefulPartitionedCall�
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65756o
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
#__inference_signature_wrapper_66013
model_input!
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
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *)
f$R"
 __inference__wrapped_model_64835o
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
StatefulPartitionedCallStatefulPartitionedCall:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�
�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026

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
�
h
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921

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
�	
�
__inference_loss_fn_6_66875R
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
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_65944
model_input%
model_65885: 
model_65887: %
model_65889: @
model_65891:@&
model_65893:@�
model_65895:	�'
model_65897:��
model_65899:	�'
model_65901:��
model_65903:	�
model_65905:
��
model_65907:	�
model_65909:	�
model_65911:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�model/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallmodel_inputmodel_65885model_65887model_65889model_65891model_65893model_65895model_65897model_65899model_65901model_65903model_65905model_65907model_65909model_65911*
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65326�
 Sigma_Activation/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65885*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65889*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65893*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65897*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65901*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65905* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65909*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036

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
�
h
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867

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
�
N
2__inference_dense_activation_1_layer_call_fn_66808

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036`
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
�
h
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_66703

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
�
�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910

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
�
�
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964

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
�
h
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_66736

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
��
�#
!__inference__traced_restore_67238
file_prefix>
$assignvariableop_conv_layer_0_kernel: 2
$assignvariableop_1_conv_layer_0_bias: @
&assignvariableop_2_conv_layer_1_kernel: @2
$assignvariableop_3_conv_layer_1_bias:@A
&assignvariableop_4_conv_layer_2_kernel:@�3
$assignvariableop_5_conv_layer_2_bias:	�B
&assignvariableop_6_conv_layer_3_kernel:��3
$assignvariableop_7_conv_layer_3_bias:	�B
&assignvariableop_8_conv_layer_4_kernel:��3
$assignvariableop_9_conv_layer_4_bias:	�<
(assignvariableop_10_dense_layer_0_kernel:
��5
&assignvariableop_11_dense_layer_0_bias:	�;
(assignvariableop_12_dense_layer_1_kernel:	�4
&assignvariableop_13_dense_layer_1_bias:$
assignvariableop_14_beta_1: $
assignvariableop_15_beta_2: #
assignvariableop_16_decay: +
!assignvariableop_17_learning_rate: '
assignvariableop_18_adam_iter:	 %
assignvariableop_19_total_3: %
assignvariableop_20_count_3: %
assignvariableop_21_total_2: %
assignvariableop_22_count_2: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: #
assignvariableop_25_total: #
assignvariableop_26_count: H
.assignvariableop_27_adam_conv_layer_0_kernel_m: :
,assignvariableop_28_adam_conv_layer_0_bias_m: H
.assignvariableop_29_adam_conv_layer_1_kernel_m: @:
,assignvariableop_30_adam_conv_layer_1_bias_m:@I
.assignvariableop_31_adam_conv_layer_2_kernel_m:@�;
,assignvariableop_32_adam_conv_layer_2_bias_m:	�J
.assignvariableop_33_adam_conv_layer_3_kernel_m:��;
,assignvariableop_34_adam_conv_layer_3_bias_m:	�J
.assignvariableop_35_adam_conv_layer_4_kernel_m:��;
,assignvariableop_36_adam_conv_layer_4_bias_m:	�C
/assignvariableop_37_adam_dense_layer_0_kernel_m:
��<
-assignvariableop_38_adam_dense_layer_0_bias_m:	�B
/assignvariableop_39_adam_dense_layer_1_kernel_m:	�;
-assignvariableop_40_adam_dense_layer_1_bias_m:H
.assignvariableop_41_adam_conv_layer_0_kernel_v: :
,assignvariableop_42_adam_conv_layer_0_bias_v: H
.assignvariableop_43_adam_conv_layer_1_kernel_v: @:
,assignvariableop_44_adam_conv_layer_1_bias_v:@I
.assignvariableop_45_adam_conv_layer_2_kernel_v:@�;
,assignvariableop_46_adam_conv_layer_2_bias_v:	�J
.assignvariableop_47_adam_conv_layer_3_kernel_v:��;
,assignvariableop_48_adam_conv_layer_3_bias_v:	�J
.assignvariableop_49_adam_conv_layer_4_kernel_v:��;
,assignvariableop_50_adam_conv_layer_4_bias_v:	�C
/assignvariableop_51_adam_dense_layer_0_kernel_v:
��<
-assignvariableop_52_adam_dense_layer_0_bias_v:	�B
/assignvariableop_53_adam_dense_layer_1_kernel_v:	�;
-assignvariableop_54_adam_dense_layer_1_bias_v:
identity_56��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
value�B�8B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*�
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp$assignvariableop_conv_layer_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv_layer_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_conv_layer_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_conv_layer_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_conv_layer_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv_layer_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_conv_layer_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_conv_layer_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_conv_layer_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_conv_layer_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_dense_layer_0_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp&assignvariableop_11_dense_layer_0_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp(assignvariableop_12_dense_layer_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_dense_layer_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_3Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_3Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_conv_layer_0_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_conv_layer_0_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_conv_layer_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_conv_layer_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_conv_layer_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_conv_layer_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_conv_layer_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv_layer_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_conv_layer_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv_layer_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp/assignvariableop_37_adam_dense_layer_0_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp-assignvariableop_38_adam_dense_layer_0_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp/assignvariableop_39_adam_dense_layer_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp-assignvariableop_40_adam_dense_layer_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_conv_layer_0_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_conv_layer_0_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_conv_layer_1_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_conv_layer_1_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_conv_layer_2_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_conv_layer_2_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_conv_layer_3_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_conv_layer_3_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp.assignvariableop_49_adam_conv_layer_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_conv_layer_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_dense_layer_0_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp-assignvariableop_52_adam_dense_layer_0_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp/assignvariableop_53_adam_dense_layer_1_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp-assignvariableop_54_adam_dense_layer_1_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_56IdentityIdentity_55:output:0^NoOp_1*
T0*
_output_shapes
: �	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_56Identity_56:output:0*�
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_65882
model_input%
model_65823: 
model_65825: %
model_65827: @
model_65829:@&
model_65831:@�
model_65833:	�'
model_65835:��
model_65837:	�'
model_65839:��
model_65841:	�
model_65843:
��
model_65845:	�
model_65847:	�
model_65849:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�model/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallmodel_inputmodel_65823model_65825model_65827model_65829model_65831model_65833model_65835model_65837model_65839model_65841model_65843model_65845model_65847model_65849*
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65067�
 Sigma_Activation/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65823*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65827*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65831*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65835*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65839*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65843* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65847*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�
�
*__inference_sequential_layer_call_fn_65653
model_input!
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmodel_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65622o
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
StatefulPartitionedCallStatefulPartitionedCall:] Y
0
_output_shapes
:���������_�
%
_user_specified_namemodel_input
�l
�

@__inference_model_layer_call_and_return_conditional_losses_65067

inputs,
conv_layer_0_64857:  
conv_layer_0_64859: ,
conv_layer_1_64884: @ 
conv_layer_1_64886:@-
conv_layer_2_64911:@�!
conv_layer_2_64913:	�.
conv_layer_3_64938:��!
conv_layer_3_64940:	�.
conv_layer_4_64965:��!
conv_layer_4_64967:	�'
dense_layer_0_65000:
��"
dense_layer_0_65002:	�&
dense_layer_1_65027:	�!
dense_layer_1_65029:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputsconv_layer_0_64857conv_layer_0_64859*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_64884conv_layer_1_64886*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_64911conv_layer_2_64913*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_64938conv_layer_3_64940*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_64965conv_layer_4_64967*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_64983�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_65000dense_layer_0_65002*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_65027dense_layer_1_65029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_64857*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_64884*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_64911*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_64938*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_64965*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_65000* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_65027*
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
�
M
1__inference_conv_activation_4_layer_call_fn_66731

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975i
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
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_66693

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
�	
�
__inference_loss_fn_2_66839Y
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
�l
�

@__inference_model_layer_call_and_return_conditional_losses_65465	
input,
conv_layer_0_65393:  
conv_layer_0_65395: ,
conv_layer_1_65399: @ 
conv_layer_1_65401:@-
conv_layer_2_65405:@�!
conv_layer_2_65407:	�.
conv_layer_3_65411:��!
conv_layer_3_65413:	�.
conv_layer_4_65417:��!
conv_layer_4_65419:	�'
dense_layer_0_65424:
��"
dense_layer_0_65426:	�&
dense_layer_1_65430:	�!
dense_layer_1_65432:
identity��$conv_layer_0/StatefulPartitionedCall�5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_1/StatefulPartitionedCall�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_2/StatefulPartitionedCall�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_3/StatefulPartitionedCall�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�$conv_layer_4/StatefulPartitionedCall�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_0/StatefulPartitionedCall�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�%dense_layer_1/StatefulPartitionedCall�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�
$conv_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputconv_layer_0_65393conv_layer_0_65395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������/� *$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_64856�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_64867�
$conv_layer_1/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_0/PartitionedCall:output:0conv_layer_1_65399conv_layer_1_65401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������G@*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894�
$conv_layer_2/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_1/PartitionedCall:output:0conv_layer_2_65405conv_layer_2_65407*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������#�*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_64921�
$conv_layer_3/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_2/PartitionedCall:output:0conv_layer_3_65411conv_layer_3_65413*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948�
$conv_layer_4/StatefulPartitionedCallStatefulPartitionedCall*conv_activation_3/PartitionedCall:output:0conv_layer_4_65417conv_layer_4_65419*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_64964�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_64983�
%dense_layer_0/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_layer_0_65424dense_layer_0_65426*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_64999�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010�
%dense_layer_1/StatefulPartitionedCallStatefulPartitionedCall+dense_activation_0/PartitionedCall:output:0dense_layer_1_65430dense_layer_1_65432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Q
fLRJ
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_65026�
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *V
fQRO
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_65036�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_0_65393*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_1_65399*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_2_65405*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_3_65411*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpconv_layer_4_65417*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_0_65424* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_layer_1_65430*
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
�
h
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948

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
�
h
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_66604

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
__inference_loss_fn_1_66830X
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
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_66747

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
�	
�
__inference_loss_fn_5_66866S
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
�
�
E__inference_sequential_layer_call_and_return_conditional_losses_66293

inputsK
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
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�)model/conv_layer_0/BiasAdd/ReadVariableOp�(model/conv_layer_0/Conv2D/ReadVariableOp�)model/conv_layer_1/BiasAdd/ReadVariableOp�(model/conv_layer_1/Conv2D/ReadVariableOp�)model/conv_layer_2/BiasAdd/ReadVariableOp�(model/conv_layer_2/Conv2D/ReadVariableOp�)model/conv_layer_3/BiasAdd/ReadVariableOp�(model/conv_layer_3/Conv2D/ReadVariableOp�)model/conv_layer_4/BiasAdd/ReadVariableOp�(model/conv_layer_4/Conv2D/ReadVariableOp�*model/dense_layer_0/BiasAdd/ReadVariableOp�)model/dense_layer_0/MatMul/ReadVariableOp�*model/dense_layer_1/BiasAdd/ReadVariableOp�)model/dense_layer_1/MatMul/ReadVariableOp�
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
:���������u
$Sigma_Activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        w
&Sigma_Activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����w
&Sigma_Activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
Sigma_Activation/strided_sliceStridedSlice$model/dense_layer_1/BiasAdd:output:0-Sigma_Activation/strided_slice/stack:output:0/Sigma_Activation/strided_slice/stack_1:output:0/Sigma_Activation/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
ellipsis_maskw
&Sigma_Activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����y
(Sigma_Activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        y
(Sigma_Activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
 Sigma_Activation/strided_slice_1StridedSlice$model/dense_layer_1/BiasAdd:output:0/Sigma_Activation/strided_slice_1/stack:output:01Sigma_Activation/strided_slice_1/stack_1:output:01Sigma_Activation/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
ellipsis_mask*
end_mask�
Sigma_Activation/SoftplusSoftplus)Sigma_Activation/strided_slice_1:output:0*
T0*'
_output_shapes
:���������g
Sigma_Activation/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
����������
Sigma_Activation/concatConcatV2'Sigma_Activation/strided_slice:output:0'Sigma_Activation/Softplus:activations:0%Sigma_Activation/concat/axis:output:0*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*^model/conv_layer_0/BiasAdd/ReadVariableOp)^model/conv_layer_0/Conv2D/ReadVariableOp*^model/conv_layer_1/BiasAdd/ReadVariableOp)^model/conv_layer_1/Conv2D/ReadVariableOp*^model/conv_layer_2/BiasAdd/ReadVariableOp)^model/conv_layer_2/Conv2D/ReadVariableOp*^model/conv_layer_3/BiasAdd/ReadVariableOp)^model/conv_layer_3/Conv2D/ReadVariableOp*^model/conv_layer_4/BiasAdd/ReadVariableOp)^model/conv_layer_4/Conv2D/ReadVariableOp+^model/dense_layer_0/BiasAdd/ReadVariableOp*^model/dense_layer_0/MatMul/ReadVariableOp+^model/dense_layer_1/BiasAdd/ReadVariableOp*^model/dense_layer_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2V
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
�
*__inference_sequential_layer_call_fn_66074

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
identity��StatefulPartitionedCall�
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65622o
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
�
�
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_64883

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
h
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_64975

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
�
�
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_64937

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
�
M
1__inference_conv_activation_3_layer_call_fn_66698

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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *U
fPRN
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_64948i
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
�
�
%__inference_model_layer_call_fn_66387

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65326o
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
�
i
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_65010

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
�
G
0__inference_Sigma_Activation_layer_call_fn_66556
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
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591`
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
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_66726

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
�t
�
@__inference_model_layer_call_and_return_conditional_losses_66469

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
�	
�
__inference_loss_fn_4_66857Z
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
�
h
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_64894

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
�;
�
E__inference_sequential_layer_call_and_return_conditional_losses_65756

inputs%
model_65697: 
model_65699: %
model_65701: @
model_65703:@&
model_65705:@�
model_65707:	�'
model_65709:��
model_65711:	�'
model_65713:��
model_65715:	�
model_65717:
��
model_65719:	�
model_65721:	�
model_65723:
identity��5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp�5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp�6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp�model/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_65697model_65699model_65701model_65703model_65705model_65707model_65709model_65711model_65713model_65715model_65717model_65719model_65721model_65723*
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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_65326�
 Sigma_Activation/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *T
fORM
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_65591�
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65697*&
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
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65701*&
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
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65705*'
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
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65709*(
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
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65713*(
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
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65717* 
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
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmodel_65721*
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
:����������
NoOpNoOp6^conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp6^conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp7^dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:���������_�: : : : : : : : : : : : : : 2n
5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_3/kernel/Regularizer/L2Loss/ReadVariableOp2n
5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp5conv_layer_4/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2p
6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp6dense_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:X T
0
_output_shapes
:���������_�
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_66821X
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
�
i
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_66812

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
�
�
,__inference_conv_layer_2_layer_call_fn_66646

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
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_64910x
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
�
�
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_66803

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
�
h
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_66637

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
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
L
model_input=
serving_default_model_input:0���������_�D
Sigma_Activation0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer_with_weights-4
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_network
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513"
trackable_list_wrapper
�
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513"
trackable_list_wrapper
 "
trackable_list_wrapper
�
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
;trace_0
<trace_1
=trace_2
>trace_32�
*__inference_sequential_layer_call_fn_65653
*__inference_sequential_layer_call_fn_66074
*__inference_sequential_layer_call_fn_66107
*__inference_sequential_layer_call_fn_65820�
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
 z;trace_0z<trace_1z=trace_2z>trace_3
�
?trace_0
@trace_1
Atrace_2
Btrace_32�
E__inference_sequential_layer_call_and_return_conditional_losses_66200
E__inference_sequential_layer_call_and_return_conditional_losses_66293
E__inference_sequential_layer_call_and_return_conditional_losses_65882
E__inference_sequential_layer_call_and_return_conditional_losses_65944�
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
 z?trace_0z@trace_1zAtrace_2zBtrace_3
�B�
 __inference__wrapped_model_64835model_input"�
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

Cbeta_1

Dbeta_2
	Edecay
Flearning_rate
Giter(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�2m�3m�4m�5m�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�2v�3v�4v�5v�"
	optimizer
,
Hserving_default"
signature_map
"
_tf_keras_input_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

(kernel
)bias
 O_jit_compiled_convolution_op"
_tf_keras_layer
�
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

*kernel
+bias
 \_jit_compiled_convolution_op"
_tf_keras_layer
�
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

,kernel
-bias
 i_jit_compiled_convolution_op"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

.kernel
/bias
 v_jit_compiled_convolution_op"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
_tf_keras_layer
�
}	variables
~trainable_variables
regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

0kernel
1bias
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

2kernel
3bias"
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

4kernel
5bias"
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
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513"
trackable_list_wrapper
�
(0
)1
*2
+3
,4
-5
.6
/7
08
19
210
311
412
513"
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
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
%__inference_model_layer_call_fn_65098
%__inference_model_layer_call_fn_66354
%__inference_model_layer_call_fn_66387
%__inference_model_layer_call_fn_65390�
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
@__inference_model_layer_call_and_return_conditional_losses_66469
@__inference_model_layer_call_and_return_conditional_losses_66551
@__inference_model_layer_call_and_return_conditional_losses_65465
@__inference_model_layer_call_and_return_conditional_losses_65540�
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
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_Sigma_Activation_layer_call_fn_66556�
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
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_66571�
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
 "
trackable_list_wrapper
.
0
1"
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
*__inference_sequential_layer_call_fn_65653model_input"�
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
*__inference_sequential_layer_call_fn_66074inputs"�
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
*__inference_sequential_layer_call_fn_66107inputs"�
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
*__inference_sequential_layer_call_fn_65820model_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_66200inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_66293inputs"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_65882model_input"�
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
E__inference_sequential_layer_call_and_return_conditional_losses_65944model_input"�
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
#__inference_signature_wrapper_66013model_input"�
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
(0
)1"
trackable_list_wrapper
.
(0
)1"
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
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_0_layer_call_fn_66580�
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
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_66594�
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
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_0_layer_call_fn_66599�
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
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_66604�
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
*0
+1"
trackable_list_wrapper
.
*0
+1"
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
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_1_layer_call_fn_66613�
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
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_66627�
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
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_1_layer_call_fn_66632�
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
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_66637�
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
,0
-1"
trackable_list_wrapper
.
,0
-1"
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
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_2_layer_call_fn_66646�
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
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_66660�
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
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_2_layer_call_fn_66665�
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
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_66670�
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
.0
/1"
trackable_list_wrapper
.
.0
/1"
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
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_3_layer_call_fn_66679�
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
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_66693�
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
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_conv_activation_3_layer_call_fn_66698�
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
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_66703�
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
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
}	variables
~trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_conv_layer_4_layer_call_fn_66712�
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
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_66726�
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
1__inference_conv_activation_4_layer_call_fn_66731�
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
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_66736�
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
'__inference_flatten_layer_call_fn_66741�
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
B__inference_flatten_layer_call_and_return_conditional_losses_66747�
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
20
31"
trackable_list_wrapper
.
20
31"
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
-__inference_dense_layer_0_layer_call_fn_66756�
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
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_66770�
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
2__inference_dense_activation_0_layer_call_fn_66775�
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
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_66780�
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
40
51"
trackable_list_wrapper
.
40
51"
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
-__inference_dense_layer_1_layer_call_fn_66789�
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
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_66803�
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
2__inference_dense_activation_1_layer_call_fn_66808�
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
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_66812�
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
__inference_loss_fn_0_66821�
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
__inference_loss_fn_1_66830�
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
__inference_loss_fn_2_66839�
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
__inference_loss_fn_3_66848�
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
__inference_loss_fn_4_66857�
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
__inference_loss_fn_5_66866�
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
__inference_loss_fn_6_66875�
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
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_model_layer_call_fn_65098input"�
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
%__inference_model_layer_call_fn_66354inputs"�
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
%__inference_model_layer_call_fn_66387inputs"�
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
%__inference_model_layer_call_fn_65390input"�
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
@__inference_model_layer_call_and_return_conditional_losses_66469inputs"�
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
@__inference_model_layer_call_and_return_conditional_losses_66551inputs"�
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
@__inference_model_layer_call_and_return_conditional_losses_65465input"�
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
@__inference_model_layer_call_and_return_conditional_losses_65540input"�
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
0__inference_Sigma_Activation_layer_call_fn_66556x"�
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
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_66571x"�
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
,__inference_conv_layer_0_layer_call_fn_66580inputs"�
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
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_66594inputs"�
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
1__inference_conv_activation_0_layer_call_fn_66599inputs"�
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
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_66604inputs"�
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
,__inference_conv_layer_1_layer_call_fn_66613inputs"�
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
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_66627inputs"�
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
1__inference_conv_activation_1_layer_call_fn_66632inputs"�
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
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_66637inputs"�
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
,__inference_conv_layer_2_layer_call_fn_66646inputs"�
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
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_66660inputs"�
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
1__inference_conv_activation_2_layer_call_fn_66665inputs"�
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
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_66670inputs"�
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
,__inference_conv_layer_3_layer_call_fn_66679inputs"�
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
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_66693inputs"�
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
1__inference_conv_activation_3_layer_call_fn_66698inputs"�
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
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_66703inputs"�
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
,__inference_conv_layer_4_layer_call_fn_66712inputs"�
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
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_66726inputs"�
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
1__inference_conv_activation_4_layer_call_fn_66731inputs"�
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
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_66736inputs"�
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
'__inference_flatten_layer_call_fn_66741inputs"�
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
B__inference_flatten_layer_call_and_return_conditional_losses_66747inputs"�
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
-__inference_dense_layer_0_layer_call_fn_66756inputs"�
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
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_66770inputs"�
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
2__inference_dense_activation_0_layer_call_fn_66775inputs"�
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
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_66780inputs"�
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
-__inference_dense_layer_1_layer_call_fn_66789inputs"�
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
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_66803inputs"�
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
2__inference_dense_activation_1_layer_call_fn_66808inputs"�
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
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_66812inputs"�
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
__inference_loss_fn_0_66821"�
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
__inference_loss_fn_1_66830"�
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
__inference_loss_fn_2_66839"�
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
__inference_loss_fn_3_66848"�
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
__inference_loss_fn_4_66857"�
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
__inference_loss_fn_5_66866"�
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
__inference_loss_fn_6_66875"�
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
K__inference_Sigma_Activation_layer_call_and_return_conditional_losses_66571S*�'
 �
�
x���������
� "%�"
�
0���������
� z
0__inference_Sigma_Activation_layer_call_fn_66556F*�'
 �
�
x���������
� "�����������
 __inference__wrapped_model_64835�()*+,-./012345=�:
3�0
.�+
model_input���������_�
� "C�@
>
Sigma_Activation*�'
Sigma_Activation����������
L__inference_conv_activation_0_layer_call_and_return_conditional_losses_66604j8�5
.�+
)�&
inputs���������/� 
� ".�+
$�!
0���������/� 
� �
1__inference_conv_activation_0_layer_call_fn_66599]8�5
.�+
)�&
inputs���������/� 
� "!����������/� �
L__inference_conv_activation_1_layer_call_and_return_conditional_losses_66637h7�4
-�*
(�%
inputs���������G@
� "-�*
#� 
0���������G@
� �
1__inference_conv_activation_1_layer_call_fn_66632[7�4
-�*
(�%
inputs���������G@
� " ����������G@�
L__inference_conv_activation_2_layer_call_and_return_conditional_losses_66670j8�5
.�+
)�&
inputs���������#�
� ".�+
$�!
0���������#�
� �
1__inference_conv_activation_2_layer_call_fn_66665]8�5
.�+
)�&
inputs���������#�
� "!����������#��
L__inference_conv_activation_3_layer_call_and_return_conditional_losses_66703j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
1__inference_conv_activation_3_layer_call_fn_66698]8�5
.�+
)�&
inputs����������
� "!������������
L__inference_conv_activation_4_layer_call_and_return_conditional_losses_66736j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
1__inference_conv_activation_4_layer_call_fn_66731]8�5
.�+
)�&
inputs����������
� "!������������
G__inference_conv_layer_0_layer_call_and_return_conditional_losses_66594n()8�5
.�+
)�&
inputs���������_�
� ".�+
$�!
0���������/� 
� �
,__inference_conv_layer_0_layer_call_fn_66580a()8�5
.�+
)�&
inputs���������_�
� "!����������/� �
G__inference_conv_layer_1_layer_call_and_return_conditional_losses_66627m*+8�5
.�+
)�&
inputs���������/� 
� "-�*
#� 
0���������G@
� �
,__inference_conv_layer_1_layer_call_fn_66613`*+8�5
.�+
)�&
inputs���������/� 
� " ����������G@�
G__inference_conv_layer_2_layer_call_and_return_conditional_losses_66660m,-7�4
-�*
(�%
inputs���������G@
� ".�+
$�!
0���������#�
� �
,__inference_conv_layer_2_layer_call_fn_66646`,-7�4
-�*
(�%
inputs���������G@
� "!����������#��
G__inference_conv_layer_3_layer_call_and_return_conditional_losses_66693n./8�5
.�+
)�&
inputs���������#�
� ".�+
$�!
0����������
� �
,__inference_conv_layer_3_layer_call_fn_66679a./8�5
.�+
)�&
inputs���������#�
� "!������������
G__inference_conv_layer_4_layer_call_and_return_conditional_losses_66726n018�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
,__inference_conv_layer_4_layer_call_fn_66712a018�5
.�+
)�&
inputs����������
� "!������������
M__inference_dense_activation_0_layer_call_and_return_conditional_losses_66780Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_dense_activation_0_layer_call_fn_66775M0�-
&�#
!�
inputs����������
� "������������
M__inference_dense_activation_1_layer_call_and_return_conditional_losses_66812X/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
2__inference_dense_activation_1_layer_call_fn_66808K/�,
%�"
 �
inputs���������
� "�����������
H__inference_dense_layer_0_layer_call_and_return_conditional_losses_66770^230�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
-__inference_dense_layer_0_layer_call_fn_66756Q230�-
&�#
!�
inputs����������
� "������������
H__inference_dense_layer_1_layer_call_and_return_conditional_losses_66803]450�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
-__inference_dense_layer_1_layer_call_fn_66789P450�-
&�#
!�
inputs����������
� "�����������
B__inference_flatten_layer_call_and_return_conditional_losses_66747b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
'__inference_flatten_layer_call_fn_66741U8�5
.�+
)�&
inputs����������
� "�����������:
__inference_loss_fn_0_66821(�

� 
� "� :
__inference_loss_fn_1_66830*�

� 
� "� :
__inference_loss_fn_2_66839,�

� 
� "� :
__inference_loss_fn_3_66848.�

� 
� "� :
__inference_loss_fn_4_668570�

� 
� "� :
__inference_loss_fn_5_668662�

� 
� "� :
__inference_loss_fn_6_668754�

� 
� "� �
@__inference_model_layer_call_and_return_conditional_losses_65465x()*+,-./012345?�<
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
@__inference_model_layer_call_and_return_conditional_losses_65540x()*+,-./012345?�<
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
@__inference_model_layer_call_and_return_conditional_losses_66469y()*+,-./012345@�=
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
@__inference_model_layer_call_and_return_conditional_losses_66551y()*+,-./012345@�=
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
%__inference_model_layer_call_fn_65098k()*+,-./012345?�<
5�2
(�%
input���������_�
p 

 
� "�����������
%__inference_model_layer_call_fn_65390k()*+,-./012345?�<
5�2
(�%
input���������_�
p

 
� "�����������
%__inference_model_layer_call_fn_66354l()*+,-./012345@�=
6�3
)�&
inputs���������_�
p 

 
� "�����������
%__inference_model_layer_call_fn_66387l()*+,-./012345@�=
6�3
)�&
inputs���������_�
p

 
� "�����������
E__inference_sequential_layer_call_and_return_conditional_losses_65882~()*+,-./012345E�B
;�8
.�+
model_input���������_�
p 

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_65944~()*+,-./012345E�B
;�8
.�+
model_input���������_�
p

 
� "%�"
�
0���������
� �
E__inference_sequential_layer_call_and_return_conditional_losses_66200y()*+,-./012345@�=
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
E__inference_sequential_layer_call_and_return_conditional_losses_66293y()*+,-./012345@�=
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
*__inference_sequential_layer_call_fn_65653q()*+,-./012345E�B
;�8
.�+
model_input���������_�
p 

 
� "�����������
*__inference_sequential_layer_call_fn_65820q()*+,-./012345E�B
;�8
.�+
model_input���������_�
p

 
� "�����������
*__inference_sequential_layer_call_fn_66074l()*+,-./012345@�=
6�3
)�&
inputs���������_�
p 

 
� "�����������
*__inference_sequential_layer_call_fn_66107l()*+,-./012345@�=
6�3
)�&
inputs���������_�
p

 
� "�����������
#__inference_signature_wrapper_66013�()*+,-./012345L�I
� 
B�?
=
model_input.�+
model_input���������_�"C�@
>
Sigma_Activation*�'
Sigma_Activation���������