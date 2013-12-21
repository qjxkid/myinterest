#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

#define LIMIT 1
#define INLIMIT 2
#define BOUNDER 20
#define THRESHOLD 100
#define REC_SIZE 8
#define REC_THRESHOLD 0.5

int SimpleInteresting(string input,string output,int flag)
{
	Mat image=imread(input,1);
	Mat ori_image=image.clone();
	Mat gray_image=imread(input,0);
	Mat gray_image_white=gray_image.clone();
	Mat subdst=gray_image.clone();
	if(image.empty())
	{
		cout<<"Couldn't read "<<input<<endl;
		return 1;
	}

	floodFill(gray_image,Point(0,0),Scalar(0),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image,Point(0,gray_image.rows-1),Scalar(0),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image,Point(gray_image.cols-1,0),Scalar(0),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image,Point(gray_image.cols-1,gray_image.rows-1),Scalar(0),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image_white,Point(0,0),Scalar(255),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image_white,Point(0,gray_image_white.rows-1),Scalar(255),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image_white,Point(gray_image_white.cols-1,0),Scalar(255),0,Scalar(LIMIT),Scalar(LIMIT));
	floodFill(gray_image_white,Point(gray_image_white.cols-1,gray_image_white.rows-1),Scalar(255),0,Scalar(LIMIT),Scalar(LIMIT));

	subtract(gray_image_white,gray_image,subdst);

	int threshold_red=0,threshold_green=0,threshold_blue=0;
	int total_num=0;
	for(int i=0;i<subdst.rows;i++)
	{
		for(int j=0;j<subdst.cols;j++)
		{
			if(subdst.at<unsigned char>(i,j)==255)
			{
				threshold_blue+=ori_image.at<Vec3b>(i,j)[0];
				threshold_green+=ori_image.at<Vec3b>(i,j)[1];
				threshold_red+=ori_image.at<Vec3b>(i,j)[2];
				total_num++;
			}
		}
	}
	threshold_blue=((double)threshold_blue/total_num)*REC_SIZE*REC_SIZE;
	threshold_green=((double)threshold_green/total_num)*REC_SIZE*REC_SIZE;
	threshold_red=((double)threshold_red/total_num)*REC_SIZE*REC_SIZE;

	Mat kernel=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
	dilate(subdst,subdst,kernel,Point(-1,-1));
	medianBlur(subdst,subdst,11);

	Mat sum(subdst.rows+1,subdst.cols+1,CV_32SC1);
	integral(subdst,sum);

	int SIZE;
	if(subdst.rows>subdst.cols)
	{
		SIZE=subdst.rows/20;
	}
	else
	{
		SIZE=subdst.cols/20;
	}
	int step=SIZE/5;
	int times=0;
	Mat pre_dst=subdst.clone();
	for(int i=0;i+SIZE<sum.rows;i+=step)
	{
		for(int j=0;j+SIZE<sum.cols;j+=step)
		{
			int temp=(sum.at<unsigned int>(i,j)+ \
				sum.at<unsigned int>(i+SIZE,j+SIZE)- \
				sum.at<unsigned int>(i,j+SIZE)- \
				sum.at<unsigned int>(i+SIZE,j));

			if(temp==0)
			{
				floodFill(pre_dst,Point(j+SIZE/2,i+SIZE/2),Scalar(255));
				integral(pre_dst,sum);
			}
		}
	}
	
	subtract(pre_dst,subdst,subdst);
	subdst=~subdst;

	string path_png=string(output);

	vector<Mat> png_img;
	Mat not_binmask;
	not_binmask.create(subdst.size(),CV_8UC1);
	not_binmask=~subdst;

	image.setTo(Scalar(255,255,255),subdst);
	split(image,png_img);
	Mat alpha(image.size(),CV_8UC1,Scalar(0));
	alpha.setTo(Scalar(255),not_binmask);

	png_img.push_back(alpha);
	Mat png1(image.size(),CV_8UC4);
	merge(png_img,png1);

	//定位
	//top
	int top,bottom,left,right;
	top=0;
	left=0;
	bottom=subdst.rows-1;
	right=subdst.cols-1;
	unsigned char* p;
	bool found=false;
	for(int i=0;!found && i<subdst.rows;i++)
	{
		p=subdst.ptr<unsigned char>(i);
		int blacknum=0;
		for(int j=0;!found && j<subdst.cols;j++)
		{
			if(p[j]==0)
				blacknum++;
			if(blacknum>BOUNDER)
			{
				top=i;
				found=true;
			}
		}
	}
	//bottom
	found=false;
	for(int i=subdst.rows-1;!found && i>=0;i--)
	{
		p=subdst.ptr<unsigned char>(i);
		int blacknum=0;
		for(int j=0;!found && j<subdst.cols;j++)
		{
			if(p[j]==0)
				blacknum++;
			if(blacknum>BOUNDER)
			{
				bottom=i;
				found=true;
			}
		}
	}
	//left
	found=false;
	for(int i=0;!found && i<subdst.cols;i++)
	{
		int blacknum=0;
		for(int j=0;!found && j<subdst.rows;j++)
		{
			p=subdst.ptr<unsigned char>(j);
			if(p[i]==0)
				blacknum++;
			if(blacknum>BOUNDER)
			{
				left=i;
				found=true;
			}
		}
	}
	//right
	found=false;
	for(int i=subdst.cols-1;!found && i>=0;i--)
	{
		int blacknum=0;
		for(int j=0;!found && j<subdst.rows;j++)
		{
			p=subdst.ptr<unsigned char>(j);
			if(p[i]==0)
				blacknum++;
			if(blacknum>BOUNDER)
			{
				right=i;
				found=true;
			}
		}
	}

	if(left<0)
		left=0;
	if(top<0)
		top=0;
	Rect final_rect(left,top,right-left,bottom-top);

	Mat png_tmp=png1(final_rect).clone();
	vector<Mat> final_color;
	split(png_tmp,final_color);
	Mat ori_final0=final_color[0].clone();
	Mat ori_final1=final_color[1].clone();
	Mat ori_final2=final_color[2].clone();
	Mat red_sum(png_tmp.rows+1,png_tmp.cols+1,CV_32SC1);
	Mat green_sum(png_tmp.rows+1,png_tmp.cols+1,CV_32SC1);
	Mat blue_sum(png_tmp.rows+1,png_tmp.cols+1,CV_32SC1);
	Mat alpha_sum(png_tmp.rows+1,png_tmp.cols+1,CV_32SC1);
	integral(final_color[0],blue_sum);
	integral(final_color[1],green_sum);
	integral(final_color[2],red_sum);
	final_color[3]=~final_color[3];
	integral(final_color[3],alpha_sum);
	Mat kernel1=getStructuringElement(MORPH_RECT,Size(3,3));
	for(int i=0;i+REC_SIZE<red_sum.rows;i+=step)
	{
		for(int j=0;j+REC_SIZE<red_sum.cols;j+=step)
		{
			int temp_alpha=(alpha_sum.at<unsigned int>(i,j)+ \
				alpha_sum.at<unsigned int>(i+REC_SIZE,j+REC_SIZE)- \
				alpha_sum.at<unsigned int>(i,j+REC_SIZE)- \
				alpha_sum.at<unsigned int>(i+REC_SIZE,j));
			if(temp_alpha>0)
				continue;

			int temp_red=(red_sum.at<unsigned int>(i,j)+ \
				red_sum.at<unsigned int>(i+REC_SIZE,j+REC_SIZE)- \
				red_sum.at<unsigned int>(i,j+REC_SIZE)- \
				red_sum.at<unsigned int>(i+REC_SIZE,j));

			int temp_green=(green_sum.at<unsigned int>(i,j)+ \
				green_sum.at<unsigned int>(i+REC_SIZE,j+REC_SIZE)- \
				green_sum.at<unsigned int>(i,j+REC_SIZE)- \
				green_sum.at<unsigned int>(i+REC_SIZE,j));

			int temp_blue=(blue_sum.at<unsigned int>(i,j)+ \
				blue_sum.at<unsigned int>(i+REC_SIZE,j+REC_SIZE)- \
				blue_sum.at<unsigned int>(i,j+REC_SIZE)- \
				blue_sum.at<unsigned int>(i+REC_SIZE,j));

			int red_error=fast_abs(temp_red-threshold_red);
			int green_error=fast_abs(temp_blue-threshold_green);
			int blue_error=fast_abs(temp_blue-threshold_blue);
			int threshold=(int)REC_SIZE*REC_SIZE*REC_THRESHOLD;
			if(red_error<threshold && green_error<threshold && blue_error<threshold)
			{
				floodFill(final_color[0],Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(0),0,Scalar(INLIMIT),Scalar(INLIMIT));
				floodFill(final_color[1],Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(0),0,Scalar(INLIMIT),Scalar(INLIMIT));
				floodFill(final_color[2],Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(0),0,Scalar(INLIMIT),Scalar(INLIMIT));
				floodFill(ori_final0,Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(255),0,Scalar(INLIMIT),Scalar(INLIMIT));
				floodFill(ori_final1,Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(255),0,Scalar(INLIMIT),Scalar(INLIMIT));
				floodFill(ori_final2,Point(j+REC_SIZE/2,i+REC_SIZE/2),Scalar(255),0,Scalar(INLIMIT),Scalar(INLIMIT));

				//新增透明背景
				Mat mask0=ori_final0.clone();
				mask0=ori_final0-final_color[0];

				Mat mask1=ori_final1.clone();
				mask1=ori_final1-final_color[1];

				Mat mask2=ori_final2.clone();
				mask2=ori_final2-final_color[2];

				//重新计算背景
				mask0=mask0^mask1;
				mask0=mask0^mask2;

				dilate(mask0,mask0,kernel1,Point(-1,-1));
				medianBlur(mask0,mask0,5);

				final_color[3]=final_color[3]|mask0;
				integral(final_color[3],alpha_sum);
			}
		}
	}

	Mat final_output=png1(final_rect).clone();
	vector<Mat> output_channels;
	split(final_output,output_channels);
	output_channels.pop_back();
	output_channels.push_back(~final_color[3]);
	output_channels[0].setTo(Scalar(255),final_color[3]);
	output_channels[1].setTo(Scalar(255),final_color[3]);
	output_channels[2].setTo(Scalar(255),final_color[3]);
	merge(output_channels,final_output);

	if(flag==0)
	{
		imwrite(path_png+".jpg",ori_image(final_rect));
	}
	else
	{
		imwrite(path_png+".png",final_output);
	}

	return 0;
}

int main(int argc,char* argv[])
{
// 	unsigned char* data;
// 	if(argc<4)
// 	{
// 		cout<<"Example: ./image_roi input_image output_image_path&name flag(0 is jpg; 1 is png)"<<endl;
// 		return 1;
// 	}
// 
// 	int flag=atoi(argv[3]);
// 	return SimpleInteresting(argv[1],argv[2],flag);
//	SimpleInteresting("F:/jz/shoestest.jpg","F:/jz/shoestest2.jpg",0);
	
	SimpleInteresting("d:/jz/weddingdress2.jpg","F:/jz/weddingdress2.jpg",0);
	return 0;
}
