from bs4 import BeautifulSoup
import expanddouban
import pandas as pd
import numpy as np
import csv

class movie():

    def __init__(self, category=[], location=''):
        self.category = category
        self.location = location

    # Task 1
    def getMovieUrl(self):
        para = ','.join([x.strip() for x in self.category]) + ',' + self.location.strip()
        url = 'https://movie.douban.com/tag/#/?sort=S&range={},{}&tags={}'.format('9', '10', para)
        return url

    # Task 2
    def get_html(self):
        url = self.getMovieUrl()
        html = expanddouban.getHtml(url,loadmore=True,waittime=2)
        return html

    # Task 3
    def display_info(self):
        print("Movie=" + self.name)
        print("rate=", self.rate)
        print("location=" + self.location)
        print("category=" + self.category)
        print("info_link=" + self.info_link)
        print("cover_link=" + self.cover_link)

    # Task 4
    def getMovies(self):
        movies = []
        html = self.get_html()
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup) Soup is fine
        article = soup.find('div', id='content').find('div', class_='article').find('div', class_='article')
        # Test for article
        # print(article) #Article is Fine
        # print('------------------------------------')
        if article:
            try:
                for element in article.find_all(class_='list-wp', recursive=False):
                    # Test for element which in class='list-wp'
                    # print(element)
                    for item in element.find_all(class_='item'):
                        # print(item)
                        movie_list = {}

                        if item.find('p', recursive=False).find('span', class_='title'):
                            movie_name = item.find('p', recursive=False).find('span', class_='title').string
                            movie_list['Name'] = movie_name
                            # Test for printing the movie name
                            # print(movie_name)

                        if item.find('p', recursive=False).find('span', class_='rate'):
                            movie_rate = item.find('p', recursive=False).find('span', class_='rate').string
                            movie_list['Rate'] = movie_rate

                        movie_list['Location'] = self.location
                        movie_list['Category'] = self.category[1]

                        if item.get('href'):
                            movie_url = item.get('href')
                            movie_list['Url'] = movie_url
                            # Test for print movie_url
                            # print(movie_url)

                        if item.find('div', class_='cover-wp').find(class_='pic'):
                            movie_pic = \
                                item.find('div', class_='cover-wp').find('span', class_='pic', recursive=False).img[
                                    'src']
                            movie_list['Pic_Url'] = movie_pic
                            # Test for printing the movie pic
                            # print(movie_pic)

                        movies.append(movie_list)
            except Exception as e:
                print("Exception raised when crawling the related content in the HTML: \n", e)
        else:
            print('No Movie Contains as for the parameter provided')

        print(movies)
        return movies
    # Task 5
    def write_csv(self):
        try:
            with open('movies.csv', 'a',encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, ['Name', 'Rate', 'Location', 'Category', 'Url', 'Pic_Url'])
                writer.writeheader()
                for item in self.getMovies():
                    writer.writerow(item)
        except Exception as e:
            print(e)

    def movie_analysis(self):
        df=pd.read_csv('movies.csv')
        dx = df.drop(df[df['Name'] == 'Name'].index)
        dx.dropna(axis=0)
        dx['Rate'] = dx['Rate'].astype(float)  #convet object to float

        # Descriptive Analysis
        describle=dx[['Name', 'Rate', 'Location', 'Category']].describe()

        # Pivot Table on Location and Category
        dx_pvt = dx.pivot_table(aggfunc='mean', columns=['Location', 'Category'])

        # Pivot Table on Category
        dx_pvt_1 = dx.pivot_table(aggfunc='mean', columns=['Category'])
        # dx_pvt_1

        with open('output.txt','w',encoding='utf-8') as txt_file:
            txt_file.write('-----------------Moive Descriptive Analysis--------------------------')
            txt_file.write('\n')
            txt_file.write(str(describle))
            txt_file.write('\n')
            txt_file.write('----------------Pivot Table on Category and Location-----------------')
            txt_file.write('\n')
            txt_file.write(str(dx_pvt))
            txt_file.write('\n')
            txt_file.write('----------------Pivot Table on Category------------------------------')
            txt_file.write('\n')
            txt_file.write(str(dx_pvt_1))



if __name__ == '__main__':

    # ----------------Please uncomment code below before running ---------------------

    m_us_1 = movie(['电影','剧情'],'美国')
    m_us_1.write_csv()

    m_us_2 = movie(['电影','爱情'],'美国')
    m_us_2.write_csv()

    m_us_3=movie(['电影','科幻'],'美国')
    m_us_3.write_csv()

    m_us_4=movie(['电影','犯罪'],'美国')
    m_us_4.write_csv()

    m_us_5=movie(['电影','战争'],'美国')
    m_us_5.write_csv()

    m_us_6 = movie(['电影', '悬疑'],'美国')
    m_us_6.write_csv()

    m_cn_1 = movie(['电影','剧情'],'大陆')
    m_cn_1.write_csv()

    m_cn_2 = movie(['电影','爱情'],'大陆')
    m_cn_2.write_csv()

    m_cn_3=movie(['电影','科幻'],'大陆')
    m_cn_3.write_csv()

    m_cn_4=movie(['电影','犯罪'],'大陆')
    m_cn_4.write_csv()

    m_cn_5=movie(['电影','战争'],'大陆')
    m_cn_5.write_csv()

    m_cn_6 = movie(['电影', '悬疑'],'大陆')
    m_cn_6.write_csv()

    m_hk_1 = movie(['电影','剧情'],'香港')
    m_hk_1.write_csv()

    m_hk_2 = movie(['电影','爱情'],'香港')
    m_hk_2.write_csv()

    m_hk_3=movie(['电影','科幻'],'香港')
    m_hk_3.write_csv()

    m_hk_4=movie(['电影','犯罪'],'香港')
    m_cn_4.write_csv()

    m_hk_5=movie(['电影','战争'],'香港')
    m_hk_5.write_csv()

    m_hk_6 = movie(['电影', '悬疑'],'香港')
    m_hk_6.write_csv()

    #Task 6
    desc=movie()
    desc.movie_analysis()







