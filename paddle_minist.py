import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os

# 网页端paddle注释本行
paddle.enable_static()

BUF_SIZE=1024
BATCH_SIZE=1024

# Step1：准备数据

# (1)数据集
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=BUF_SIZE), batch_size=BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.test(), buf_size=BUF_SIZE), batch_size=BATCH_SIZE)
    
# （2）定义数据层
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# （3）获取分类器
hidden1 = fluid.layers.fc(input=image, size=200, act='relu') 
hidden2 = fluid.layers.fc(input=hidden1, size=100, act='relu') 
predict = fluid.layers.fc(input=hidden2, size=10, act='softmax') 

# （4）定义损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)  
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)

# （5）定义优化函数
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.005)  
opts = optimizer.minimize(avg_cost)


# Step2.网络配置

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 获取测试程序
test_program = fluid.default_main_program().clone(for_test=True)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# （2）告知网络传入的数据分为两部分，第一部分是image值，第二部分是label值
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# (3)展示模型训练曲线
all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]


def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()

EPOCH_NUM=20
model_save_dir = "./hand.inference.model"
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):                         #遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                                        feed=feeder.feed(data),              #给模型喂入数据
                                        fetch_list=[avg_cost, acc])          #fetch 误差、准确率  
        
        all_train_iter=all_train_iter+BATCH_SIZE
        all_train_iters.append(all_train_iter)
        
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

    # 进行测试
    test_accs = []
    test_costs = []
    #每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()): 
        test_cost, test_acc = exe.run(program=test_program, #执行训练程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差
        
       
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    test_acc = (sum(test_accs) / len(test_accs))

    print('epoch:%d, cost:%0.5f, Accuracy:%0.5f'%(pass_id, test_cost, test_acc))

draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

print ('save models to %s' % (model_save_dir))

fluid.io.save_inference_model(model_save_dir,   # 保存推理model的路径
                                  ['image'],    # 推理（inference）需要 feed 的数据
                                  [predict],    # 保存推理（inference）结果的 Variables
                                  exe)          # executor 保存 inference model
print('训练模型保存完成！')

infer_path='./images/myhandle9.png'
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

predict = []

# (4)开始预测
for i in range(0, 10):
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)
        im = Image.open("./images/myhandle" + str(i) + ".png").convert('L')
        # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
        im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32) 
        im = im / 255.0 * 2.0 - 1.0  # 归一化到【-1~1】之间

        results = infer_exe.run(program=inference_program, feed={feed_target_names[0]: im}, fetch_list=fetch_targets) 
        # 获取概率最大的label, argsort函数返回的是result数组值从小到大的索引值
        lab = np.argsort(results)
        print("图片 myhandle" + str(i) + ".png 的预测结果为: %d" % lab[0][0][-1])
        predict.append(lab[0][0][-1])

plt.subplot(1, 10, 1)
im = Image.open('./images/myhandle0.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[0]))

plt.subplot(1, 10, 2)
im = Image.open('./images/myhandle1.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[1]))

plt.subplot(1, 10, 3)
im = Image.open('./images/myhandle2.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[2]))

plt.subplot(1, 10, 4)
im = Image.open('./images/myhandle3.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[3]))

plt.subplot(1, 10, 5)
im = Image.open('./images/myhandle4.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[4]))

plt.subplot(1, 10, 6)
im = Image.open('./images/myhandle5.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[5]))

plt.subplot(1, 10, 7)
im = Image.open('./images/myhandle6.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[6]))

plt.subplot(1, 10, 8)
im = Image.open('./images/myhandle7.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[7]))

plt.subplot(1, 10, 9)
im = Image.open('./images/myhandle8.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[8]))

plt.subplot(1, 10, 10)
im = Image.open('./images/myhandle9.png')
plt.imshow(im,cmap='gray')
plt.xlabel(str(predict[9]))

plt.show()