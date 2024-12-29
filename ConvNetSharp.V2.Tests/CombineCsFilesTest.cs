using NUnit.Framework;

namespace ConvNetSharp.V2.Tests;

[TestFixture]
public class CombineCsFilesTest
{
    [SetUp]
    public void Setup()
    {
        // Delete the output file if it exists before each test run
        if (File.Exists(this._outputFilePath))
        {
            File.Delete(this._outputFilePath);
        }
    }

    private readonly string _basePath = @"E:\\Pro\\Github\\ConvNetSharp\\ConvNetSharp.V2";
    private readonly string _outputFilePath = @"E:\\Pro\\Github\\ConvNetSharp\\CombinedFiles.txt";

    [Ignore("Used to create a context file")]
    [Test]
    public void CombineAllCsFilesIntoSingleFile()
    {
        // Get all .cs files under the base path
        var csFiles = Directory.GetFiles(this._basePath, "*.cs", SearchOption.AllDirectories);

        using (var outputFile = new StreamWriter(this._outputFilePath))
        {
            foreach (var file in csFiles)
            {
                // Get relative path
                var relativePath = Path.GetRelativePath(this._basePath, file);

                // Write relative path as a header
                outputFile.WriteLine($"// --- File: {relativePath} ---\n");

                // Write file content
                var fileContent = File.ReadAllText(file);
                outputFile.WriteLine(fileContent);
                outputFile.WriteLine(); // Add a blank line for separation
            }
        }

        Console.WriteLine($"Combined file created at: {this._outputFilePath}");
    }
}