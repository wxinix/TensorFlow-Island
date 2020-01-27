using RemObjects.Elements.EUnit;

namespace TensorFlow.Island.Tests
{
    public static class Program
    {
         public Int32 Main(string[] args)
         {             
            var lTests = Discovery.DiscoverTests();
            Runner.RunTests(lTests) withListener(Runner.DefaultListener);
         }
    }
}