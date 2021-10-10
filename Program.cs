Console.InputEncoding = Encoding.UTF8;
Console.OutputEncoding = Encoding.UTF8;
ISentimentService service = new SentimentService();
try
{
    if (args.Length > 0)
    {
        switch (args[0]?.Trim())
        {
            case "train":
                Train(args);
                break;
            case "predict":
                Predict(args);
                break;
            default:
                Helper();
                break;
        }
    }
    else
    {
        Helper();
    }
}
catch (Exception ex)
{
    Console.WriteLine($"{ex.Message}\n{ex.StackTrace}");
    Helper();
}

/// <summary>
/// Thử nghiệm cảm xúc
/// </summary>
void Predict(string[] args)
{
    service.Predict(args[1], args[2]);
}

/// <summary>
/// Xây dựng và đào tạo mô hình cảm xúc với thuật toán phân loại nhị phân
/// </summary>
void Train(string[] args)
{
    if (Directory.Exists(args[1]))
    {
        string folderPath = args[1];
        string fName = $"_{DateTime.Now.Ticks}.zip";
        if (args.Length == 3)
        {
            fName = $"{args[2]}.zip";
        }
        string fPath = $"{new DirectoryInfo(folderPath).FullName}\\{fName}";

        var files = Directory.GetFiles(args[1], "*.txt", SearchOption.AllDirectories);
        if (files.Length > 0)
        {
            bool flag = true;
            List<SentimentData> list = new List<SentimentData>();
            for (int i = 0; i < files.Length; i++)
            {
                if (!flag)
                {
                    break;
                }
                var lines = File.ReadAllLines(files[i]);
                for (int x = 0; x < lines.Length; x++)
                {
                    var arr = lines[x]?.Trim().Split(new string[] { "\t", "|" }, StringSplitOptions.RemoveEmptyEntries);
                    if (arr?.Length == 2)
                    {
                        list.Add(new SentimentData { SentimentText = arr[0], Sentiment = arr[1] == "1" ? true : false });
                    }
                    else
                    {
                        Console.WriteLine($"Tập tin: {files[i]} => định dạng không đúng: dòng {x}");
                        list.Clear();
                        flag = false;
                        break;
                    }
                }
            }
            if (list.Count > 0)
            {
                try
                {
                    service.Train(list, fPath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"{ex.Message}\n{ex.StackTrace}");
                }
            }
            else
            {
                Console.WriteLine("Lỗi: Không tìm thấy dữ liệu.");
            }
        }
        else
        {
            Console.WriteLine("Lỗi: không tìm thấy các tập tin *.txt.");
        }
    }
    else
    {
        Helper();
    }
}

/// <summary>
/// Hướng dẫn sử dụng
/// </summary>
void Helper()
{
    Console.WriteLine("----- User manual -----");
    Console.WriteLine("Program.exe [*train] [*Đường dẫn thư mục chứa dữ liệu cảm xúc] [?Tên mô hình]");
    Console.WriteLine("Program.exe [*predict] [*Đường dẫn mô hình đã đào tạo (.zip)] [*Bình luận]");
}