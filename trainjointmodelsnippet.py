settings = modeltypes[i]

if settings[1][0] == 'b':
    textparams['bidir'] = True
    textparams['rnnType'] = settings[1][2:]
else:
    textparams['bidir'] = False
    textparams['rnnType'] = settings[1]

model = jointModel(settings[0],'rnn',{'tree':treemodels[settings[0]],'text':textmodels[settings[1]]},textparams,treeparams,X_text,y,device)
model = model.to(device)
modelname = settings[0]+'_'+settings[1]
print(modelname)

if settings[0][0] == 't':
    trainTemporalModel(model,modelname)
else:
    trainModel(model,modelname)