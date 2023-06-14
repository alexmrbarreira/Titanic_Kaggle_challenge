from commons import *

# ======================================================== 
# Load data
# ======================================================== 

df       = pd.read_csv('data_store/data_modified_train.csv')
df_train = pd.read_csv('data_store/data_my_train.csv')
df_valid = pd.read_csv('data_store/data_my_valid.csv')

print ('')
print ('Total percentage of survivals in the total training set:', df['Survived'].sum() / df.shape[0])
print ('Total percentage of survivals in the modified  training set:', df_train['Survived'].sum() / df_train.shape[0])
print ('Total percentage of survivals in the modified validation set:', df_valid['Survived'].sum() / df_valid.shape[0])
print ('')

# ======================================================== 
# Plot probabilities to survive
# ========================================================

def plot_dist(df, feature, ln, ls, c, ls2, c2, nbins, addlabel, addbar):
    bin_vals, prob = get_prob_inbins(feature, nbins, df)
    bin_vals, dist = get_dist_inbins(feature, nbins, df)
    width = bin_vals[1] - bin_vals[0]
    if (addlabel):
        plt.plot(bin_vals, prob, linewidth = ln, linestyle = ls , c = c, label = 'Survive \n probability')
    else:
        plt.plot(bin_vals, prob, linewidth = ln, linestyle = ls , c = c)
    if (addbar):
        if(addlabel):
            plt.bar(bin_vals, dist, color = c2, alpha = alpha_c, width = width*(9/10.), label = 'Number of \n passengers')
        else:
            plt.bar(bin_vals, dist, color = c2, alpha = alpha_c, width = width*(9/10.))
    return 0

y_min      = -0.01
label_prob = 'Survive \n probability'
label_dist = 'Passenger \n number'

fig0 = plt.figure(0, figsize=(17., 7.))
fig0.subplots_adjust(left=0.08, right=0.96, top=0.98, bottom=0.16, wspace = 0.35, hspace = 0.45)

# Ticket class  
panel = fig0.add_subplot(2,4,1)
def plot_ticket_class(df, ln, ls, c, ls2, c2, addbar):
    P1 = df[(df['Pclass'] == 1)]['Survived'].sum() / df[(df['Pclass'] == 1)].shape[0]
    P2 = df[(df['Pclass'] == 2)]['Survived'].sum() / df[(df['Pclass'] == 2)].shape[0]
    P3 = df[(df['Pclass'] == 3)]['Survived'].sum() / df[(df['Pclass'] == 3)].shape[0]
    D1 = df[(df['Pclass'] == 1)].shape[0]
    D2 = df[(df['Pclass'] == 2)].shape[0]
    D3 = df[(df['Pclass'] == 3)].shape[0]
    plt.plot([1,2,3], [P1, P2, P3]                            , linewidth = ln, linestyle = ls , c = c)
    if(addbar):
        plt.bar([1,2,3], np.array([D1, D2, D3])/max([D1, D2, D3]), color = c2, alpha = alpha_c)
    return 0
plot_ticket_class(df      , ln_def, ls_def , c_def, ls_def , c_def2, True)
plot_ticket_class(df_train, ln_def, ls_def2, c_def, ls_def2, c_def2, False)
plot_ticket_class(df_valid, ln_def, ls_def3, c_def, ls_def3, c_def2, False)
plt.xlabel(r'Ticket class' , fontsize = labelsize)
plt.ylabel(r'Distributions'          , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)

# Name characters
panel = fig0.add_subplot(2,4,2)
plot_dist(df      , 'Namechars', ln_def, ls_def , c_def, ls_def , c_def2, 10, False, True)
plot_dist(df_train, 'Namechars', ln_def, ls_def2, c_def, ls_def2, c_def2, 10, False, False)
plot_dist(df_valid, 'Namechars', ln_def, ls_def3, c_def, ls_def3, c_def2, 10, False, False)
plt.xlabel(r'Name characters' , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)

# Sex
panel = fig0.add_subplot(2,4,3)
def plot_sex(df, ln, ls, c, ls2, c2, addbar):
    P1 = df[(df['Sex_vals'] == 0)]['Survived'].sum() / df[(df['Sex_vals'] == 0)].shape[0]
    P2 = df[(df['Sex_vals'] == 1)]['Survived'].sum() / df[(df['Sex_vals'] == 1)].shape[0]
    D1 = df[(df['Sex_vals'] == 0)].shape[0]
    D2 = df[(df['Sex_vals'] == 1)].shape[0]
    plt.plot([1,2], [P1, P2]                        , linewidth = ln, linestyle = ls , c = c)
    if (addbar):
        plt.bar([1,2], np.array([D1, D2])/max([D1, D2]), color = c2, alpha = alpha_c)
plot_sex(df      , ln_def, ls_def , c_def, ls_def , c_def2, True)
plot_sex(df_train, ln_def, ls_def2, c_def, ls_def2, c_def2, False)
plot_sex(df_valid, ln_def, ls_def3, c_def, ls_def3, c_def2, False)
plt.xticks([1,2], ['Male', 'Female'], rotation = 20.)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize-2)

# Age
panel = fig0.add_subplot(2,4,4)
plot_dist(df      , 'Age', ln_def, ls_def , c_def, ls_def , c_def2, 10, False, True)
plot_dist(df_train, 'Age', ln_def, ls_def2, c_def, ls_def2, c_def2, 10, False, False)
plot_dist(df_valid, 'Age', ln_def, ls_def3, c_def, ls_def3, c_def2, 10, False, False)
plt.xlabel(r'Age' , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)

# Number of siblings/spouse
panel = fig0.add_subplot(2,4,5)
plot_dist(df      , 'SibSp', ln_def, ls_def , c_def, ls_def , c_def2, 5, True, True)
plot_dist(df_train, 'SibSp', ln_def, ls_def2, c_def, ls_def2, c_def2, 5, False, False)
plot_dist(df_valid, 'SibSp', ln_def, ls_def3, c_def, ls_def3, c_def2, 5, False, False)
plt.xlabel(r'N siblings, spouses' , fontsize = labelsize)
plt.ylabel(r'Distributions'          , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': legend_font-4}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)

# Number of parents/children
panel = fig0.add_subplot(2,4,6)
plot_dist(df      , 'Parch', ln_def, ls_def , c_def, ls_def , c_def2, 5, False, True)
plot_dist(df_train, 'Parch', ln_def, ls_def2, c_def, ls_def2, c_def2, 5, False, False)
plot_dist(df_valid, 'Parch', ln_def, ls_def3, c_def, ls_def3, c_def2, 5, False, False)
plt.xlabel(r'N parents, children' , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
plt.annotate(r'Solid: all training'    , xy = (0.30, 0.85), xycoords = 'axes fraction', c = 'k', fontsize = text_font-6)
plt.annotate(r'Dashed: $70\%$ training', xy = (0.30, 0.75), xycoords = 'axes fraction', c = 'k', fontsize = text_font-6)
plt.annotate(r'Dotted: $30\%$ training', xy = (0.30, 0.65), xycoords = 'axes fraction', c = 'k', fontsize = text_font-6)

# Fare
panel = fig0.add_subplot(2,4,7)
plot_dist(df      , 'Fare', ln_def, ls_def , c_def, ls_def , c_def2, 10, False, True)
plot_dist(df_train, 'Fare', ln_def, ls_def2, c_def, ls_def2, c_def2, 10, False, False)
plot_dist(df_valid, 'Fare', ln_def, ls_def3, c_def, ls_def3, c_def2, 10, False, False)
plt.xlabel(r'Fare' , fontsize = labelsize)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)

# Embarked
panel = fig0.add_subplot(2,4,8)
def plot_embarked(df, ln, ls, c, ls2, c2, addbar):
    P1 = df[(df['Embarked_vals'] == 0)]['Survived'].sum() / df[(df['Embarked_vals'] == 0)].shape[0]
    P2 = df[(df['Embarked_vals'] == 1)]['Survived'].sum() / df[(df['Embarked_vals'] == 1)].shape[0]
    P3 = df[(df['Embarked_vals'] == 2)]['Survived'].sum() / df[(df['Embarked_vals'] == 2)].shape[0]
    D1 = df[(df['Embarked_vals'] == 0)].shape[0]
    D2 = df[(df['Embarked_vals'] == 1)].shape[0]
    D3 = df[(df['Embarked_vals'] == 2)].shape[0]
    plt.plot([1,2,3], [P1, P2, P3]                            , linewidth = ln, linestyle = ls , c = c)
    if(addbar):
        plt.bar([1,2,3], np.array([D1, D2, D3])/max([D1, D2, D3]), color = c2, alpha = alpha_c)
    return 0
plot_embarked(df      , ln_def, ls_def , c_def, ls_def , c_def2, True)
plot_embarked(df_train, ln_def, ls_def2, c_def, ls_def2, c_def2, False)
plot_embarked(df_valid, ln_def, ls_def3, c_def, ls_def3, c_def2, False)
plt.xticks([1,2,3], ['Cherbourg', 'Queenstown', 'Southampton'], rotation = 30.)
plt.ylim(ymin = y_min)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize-6)

fig0.savefig('fig_store/fig_data_trends_distributions.png')

# ======================================================== 
# Plot covariance of selected features
# ========================================================

df_cov       = df[['Pclass', 'Namechars', 'Sex_vals', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_vals']]
df_cov       = df_cov.dropna()
array_df_cov = df_cov.values
cov_matrix   = np.cov(np.transpose(array_df_cov))
outer_diag   = np.outer( np.diag(cov_matrix), np.diag(cov_matrix))
corr_matrix  = cov_matrix/np.sqrt(outer_diag)

fig1 = plt.figure(1, figsize=(9.6, 8.))
fig1.subplots_adjust(left=0.12, right=0.94, top=0.94, bottom=0.16, wspace = 0.35, hspace = 0.45)

panel = fig1.add_subplot(1,1,1)
plt.title(r'Correlation matrix (all training data)', fontsize = title_font)
panel.set_aspect('equal')
plt.imshow(corr_matrix, cmap = 'seismic', vmin = -1, vmax = 1)
xy_ticks = range(len(corr_matrix[:,0]))
xy_ticknames = ['Ticket class', 'Name \n characters', 'Sex', 'Name', 'N siblings', 'N parents', 'Fare', 'Embarked']
plt.xticks(xy_ticks, xy_ticknames, fontsize = labelsize-6, rotation = 45.)
plt.yticks(xy_ticks, xy_ticknames, fontsize = labelsize-6, rotation = 45.)
cb = plt.colorbar()
cb.set_label(r'${\rm Cov}^{ij}/\sqrt{{\rm Cov}^{ii}{\rm Cov}^{jj}}$', fontsize = labelsize-4)
cb.ax.tick_params(labelsize=ticksize-4)
# Add number to plot
for i in range(len(corr_matrix[:,0])):
    for j in range(i):
        corr_now = round(corr_matrix[i,j], 2)
        if (corr_now >= 0):
            plt.annotate(r'$'+str(corr_now)+'$', xy = (j-0.25, i+0.1), xycoords = 'data', c = 'k', fontsize = text_font)
        else:
            plt.annotate(r'$'+str(corr_now)+'$', xy = (j-0.40, i+0.1), xycoords = 'data', c = 'k', fontsize = text_font)

fig1.savefig('fig_store/fig_data_trends_covariance.png')

plt.show()


